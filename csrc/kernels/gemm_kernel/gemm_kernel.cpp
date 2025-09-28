#include <hip/hip_runtime.h>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
using DsLayout = ck_tile::tuple<>;

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>,
                        ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool Persistent,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs</*NumDTensor = 0*/>& args, const ck_tile::stream_config& s)

{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                           GemmConfig::kPadN,
                                           GemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           GemmConfig::NumWaveGroups>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run =
        [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 CDataType,
                                                 DsLayout,
                                                 ELayout,
                                                 CDEElementWise,
                                                 UniversalGemmProblem::kBlockSize,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 GemmConfig::M_Warp,
                                                 GemmConfig::N_Warp,
                                                 GemmConfig::M_Warp_Tile,
                                                 GemmConfig::N_Warp_Tile,
                                                 GemmConfig::K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation,
                                                 GemmConfig::NumWaveGroups>>;
            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            dim3 grids;
            if constexpr(Persistent)
            {
                grids = Kernel::MaxOccupancyGridSize(s);
            }
            else
            {
                grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            }
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                          << "shape: " << GemmShape::GetName() << '\n'
                          << "problem: " << UniversalGemmProblem::GetName() << '\n'
                          << "pipeline: " << GemmPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }
            if(s.flush_cache_)
            {
                std::cout << "Flushing cache..." << std::endl;
                static constexpr ck_tile::index_t APackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
                static constexpr ck_tile::index_t BPackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

                ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{})));
                ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                    args.K, args.N, args.stride_B, is_row_major(BLayout{})));

                auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
                auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

                ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                    kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
                rotating_mem.Print();

                auto run_flush_cache = [&]() {
                    // flush icache
                    ck_tile::flush_icache();
                    // rotating mem
                    rotating_mem.Next();
                    // clear c mem
                    if(args.k_batch > 1)
                        hipGetErrorString(hipMemsetAsync(
                            args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
                };
                ave_time = ck_tile::launch_kernel_preprocess(
                    s,
                    run_flush_cache,
                    ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                        Kernel{}, grids, blocks, 0, kargs));
            }
            else
            {
                ave_time =
                    ck_tile::launch_kernel(s,
                                           ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                               Kernel{}, grids, blocks, 0, kargs));
            }
            return ave_time;
        };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}

extern "C" float gemm_fp16_fp16_fp16(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {
    
    if (persistent)
    {
        return gemm<GemmConfigComputeV3<ck_tile::half_t>,
                    GemmTypeConfig<ck_tile::half_t>::ADataType,
                    GemmTypeConfig<ck_tile::half_t>::BDataType,
                    ck_tile::tuple<>,
                    GemmTypeConfig<ck_tile::half_t>::AccDataType,
                    GemmTypeConfig<ck_tile::half_t>::CDataType,
                    ALayout, BLayout, DsLayout, CLayout,
                    true, 
                    ck_tile::element_wise::PassThrough>(args, s);
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::half_t>,
                    GemmTypeConfig<ck_tile::half_t>::ADataType,
                    GemmTypeConfig<ck_tile::half_t>::BDataType,
                    ck_tile::tuple<>,
                    GemmTypeConfig<ck_tile::half_t>::AccDataType,
                    GemmTypeConfig<ck_tile::half_t>::CDataType,
                    ALayout, BLayout, DsLayout, CLayout,
                    false,
                    ck_tile::element_wise::PassThrough>(args, s);
    }
}

extern "C" float gemm_bf16_bf16_bf16(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {
    
    if (persistent)
    {
       return gemm<GemmConfigComputeV3<ck_tile::bf16_t>,
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::ADataType, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::AccDataType, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                true,
                ck_tile::element_wise::PassThrough>(args, s); 
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::bf16_t>,
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::ADataType, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::AccDataType, 
                GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                false,
                ck_tile::element_wise::PassThrough>(args, s);
    }
}

extern "C" float gemm_fp8_fp8_fp16(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {

    if (persistent)
    {
        return gemm<GemmConfigComputeV3<ck_tile::fp8_t>, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                true,
                ck_tile::element_wise::PassThrough>(args, s);
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::fp8_t>, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                false,
                ck_tile::element_wise::PassThrough>(args, s);
    }

}

extern "C" float gemm_bf8_bf8_fp16(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {

    if (persistent)
    {
        return gemm<GemmConfigComputeV3<ck_tile::bf8_t>, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                true,
                ck_tile::element_wise::PassThrough>(args, s);
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::bf8_t>,  
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                false,
                ck_tile::element_wise::PassThrough>(args, s);
    }
    
}

extern "C" float gemm_fp16_pk_int4_fp16(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {
    
    if (persistent) 
    {
        return gemm<GemmConfigComputeV3<ck_tile::half_t>, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout,
                true,
                ck_tile::element_wise::PassThrough>(args, s);
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::half_t>, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::ADataType, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::AccDataType, 
                GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                false,
                ck_tile::element_wise::PassThrough>(args, s);
    }
}

extern "C" float gemm_int8_int8_int32(
    const ck_tile::GemmHostArgs<>& args,
    const ck_tile::stream_config& s,
    const bool persistent) {

    if (persistent) 
    {
        return gemm<GemmConfigComputeV3<ck_tile::int8_t>, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::ADataType, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::AccDataType, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                true,
                ck_tile::element_wise::PassThrough>(args, s);
    }
    else
    {
        return gemm<GemmConfigComputeV3<ck_tile::int8_t>,  
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::ADataType, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::BDataType, 
                ck_tile::tuple<>, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::AccDataType, 
                GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>::CDataType,
                ALayout, BLayout, DsLayout, CLayout, 
                false,
                ck_tile::element_wise::PassThrough>(args, s);
    }
}