// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"


namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockUniversalGemmAsBsCrWithScales
{
    private:
    // TODO: This should be in Policy - UniversalGemmPolicyBase ?
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem         = remove_cvref_t<PipelineProblem_>;
        using Policy          = remove_cvref_t<GemmPolicy_>;
        using ADataType       = remove_cvref_t<typename Problem::ADataType>;
        using BDataType       = remove_cvref_t<typename Problem::BDataType>;
        using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
        using CDataType       = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        //workgroup内的线程数 256
        static constexpr auto Scheduler     = Problem::Scheduler;
        //调度策略

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;
        //workgroup计算的A[kM,kK] B[kK,kN] C[kM,kN]维度大小

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;
        //得到WarpGemm负责计算子tile的大小

        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();
        //workgroup在kM，kN大小划分的warp数量

        using I0 = number<0>;
        using I1 = number<1>;

        static_assert(MWarp == BlockGemmShape::BlockWarps::at(I0{}),
                      "Error! WarpGemm's MWarp is not consisten with BlockGemmShape!");
        static_assert(NWarp == BlockGemmShape::BlockWarps::at(I1{}),
                      "Error! WarpGemm's NWarp is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kM == BlockGemmShape::WarpTile::at(I0{}),
                      "Error! WarpGemm's M is not consisten with BlockGemmShape!");
        static_assert(WarpGemm::kN == BlockGemmShape::WarpTile::at(I1{}),
                      "Error! WarpGemm's N is not consisten with BlockGemmShape!");

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        //一个warp负责的大小在kM上按照warp_tile大小要计算多少次
        //256/(2*32)=4 A[256,64]在M方向分为2个warp，每个warp负责[128,64] warp_tile在M方向大小是32，128/32=4
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        //256/(2*32)=4 B[64,256]在N方向分为2个warp，每个warp负责[64,128] warp_tile在N方向大小是32，128/32=4
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;
        //64/16=4 [128,64]在K方向只有1个warp，warp_tile在K方向大小是16 64/16=4

        //C矩阵[256,256]大小被分为4块[128,128]，总共4个warp负责计算
        //每块[128,128]大小由A[128,64] B[64,128]计算得到
        //每个warp一次能计算A[32,16] B[16,32] 因此在这个warp负责的[128,128]需要32x32x16大小累计计算

        static_assert(MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock,
                      "Error! Warps should cover all Block tile!");
        static_assert(NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock,
                      "Error! Warps should cover all Block tile!");

        static constexpr index_t MPerBlockPerIter = MWarp * WarpGemm::kM;
        //一次计算warp在M方向能覆盖大小 2*32=64
        static constexpr index_t NPerBlockPerIter = NWarp * WarpGemm::kN;
        //2*32=64
        static constexpr index_t KPerBlockPerIter = WarpGemm::kK;
        //16

        // Controls how many MAC clusters (MFMA blocks) we have per wave
        // Ie if
        // InterWaveSchedulingMacClusters = 1;
        // KPerBlock == 32
        // WarpGemm::kK = 8
        // Then we would group all 4 WarpGemms into single MAC cluster.
        // But if we would set InterWaveSchedulingMacClusters = 2, then we would
        // split those 4 warp gemms into two groups.
        static constexpr index_t InterWaveSchedulingMacClusters = 1;

        // should be at least equal to: WarpGemm::Impl::kABKPerLane
        static constexpr index_t KPack      = WarpGemm::kKPerThread;

        static constexpr index_t KPerThread = KIterPerWarp * WarpGemm::kKPerThread;

    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType       = remove_cvref_t<typename Traits::ADataType>;
    using BDataType       = remove_cvref_t<typename Traits::BDataType>;
    using ComputeDataType = remove_cvref_t<typename Traits::ComputeDataType>;
    using CDataType       = remove_cvref_t<typename Traits::CDataType>;

    using WarpGemm = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    //4
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    //4
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;
    //4

    static constexpr index_t MWarp = Traits::MWarp;
    //2
    static constexpr index_t NWarp = Traits::NWarp;
    //2

    static constexpr auto Scheduler = Traits::Scheduler;

    using AWarpDstr = typename WarpGemm::AWarpDstr;
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    using CWarpDstr = typename WarpGemm::CWarpDstr;
    //A,B,C在warp内部64个线程对32x32x16大小划分和访问

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;
    //A,B,C在warp内部的A/B/C张量类型，描述每个线程实际操作的数据

    static constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    //32x16
    static constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    //16x32
    static constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    //32x32

    static constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    static constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    static constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
    //0,0

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using I0 = number<0>;
    using I1 = number<1>;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread; 
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, KIterSeq>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return a_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeAScaleBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;
        // constexpr index_t KScalePerChannel = Traits::KPerBlock/128;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;
        
        constexpr auto a_scale_block_outer_dstr_encoding =
            tile_distribution_encoding<
                sequence<NWarp>,   
                tuple<sequence<MIterPerWarp, MWarp>, KIterSeq>, 
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
        
        using AScaleWarpDstrEncoding =
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<32>, 
                      sequence<1, 1>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<0, 0>>,
                sequence<2>,
                sequence<1>>;

        constexpr auto a_scale_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_scale_block_outer_dstr_encoding, AScaleWarpDstrEncoding{});

        return a_scale_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBScaleBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;
        
        constexpr auto b_scale_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<1, NWarp>, KIterSeq>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        
        using BScaleWarpDstrEncoding = 
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<1>, 
                      sequence<1, 1>>,
                tuple<sequence<2, 1>>,
                tuple<sequence<0, 0>>,
                sequence<2>,
                sequence<1>>;

        constexpr auto b_scale_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_scale_block_outer_dstr_encoding, BScaleWarpDstrEncoding{});

        return b_scale_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr index_t KPerThread     = Traits::KPerThread;
        constexpr index_t NumMacClusters = Traits::InterWaveSchedulingMacClusters;
        constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        constexpr index_t KIterInterwave = KPerInnerLoop / WarpGemm::kKPerThread;

        using KIterSeq = std::conditional_t<Scheduler == GemmPipelineScheduler::Interwave,
                                            sequence<KIterInterwave>,
                                            sequence<KIterPerWarp>>;

        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, KIterSeq>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    private:
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& warp_tile,
                                                        const WarpWindow& warp_window)
    {
        constexpr index_t UnaryOpSize = 8;
        //每次elementwise操作处理8个元素
        const element_wise::PassThroughPack8 elementwise_op{};
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto in_dstr_tensors           = load_tile(warp_window);
        //从LDS窗口（warp_window）加载tile到一个中间buffer（通常是寄存器）

        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);

        using ComputeVectorType = ComputeDataType __attribute__((ext_vector_type(UnaryOpSize)));
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            elementwise_op(warp_tile.get_thread_buffer().template get_as<ComputeVectorType>()(i),
                           in_dstr_tensors.get_thread_buffer().template get_as<pk_int4x4_t>()[i]);
        });
    }
    //将LDS中的tile数据（warp_window）加载到寄存器tile

    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Default, GemmTraits>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};
        //A/B在LDS中的tile分布策略,A/B大tile在整个workgroup内如何分配给warp、线程、循环
        //[256,64] [64,256] 每个warp负责不同的区域

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));
        //描述了A/B tile在LDS到寄存器的分布和存储方式，每个线程/warp在寄存器中要存放A/B的哪一部分数据
        //32x16,16x32

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,//描述A tile在LDS中的分布和访问方式
                                       const BSmemBlockWindow& b_block_window) //描述B tile在LDS中的分布和访问方式
        //C[256,256] A[256,64] B[64,256]
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");
            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);
                //[128,64]
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                load_interleaved_pk_type(b_warp_tile_, b_block_window);
            }
            else
            {
                load_tile(b_warp_tile_, b_block_window);
                //[64,128]
            }
            // hot loop:
            static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                //K方向上循环64/16=4
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    //M方向上循环128/32=4
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                    //[128，64]中取出[32,16]

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        //N方向循环
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                        //从[64,128]取出[16,32]

                        // read C warp tensor from C block tensor-
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        //从[128,128]取出[32,32]

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                        //写回[32,32]
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        template <typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);

            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                load_interleaved_pk_type(b_warp_tile_, b_block_window);
            }
            else
            {
                load_tile(b_warp_tile_, b_block_window);
            }
        }


        // // C += A * B
        // template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow, typename ABlockTensor, typename BBlockTensor>
        // CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
        //                                [[maybe_unused]] ASmemBlockWindow& a_block_window,
        //                                [[maybe_unused]] BSmemBlockWindow& b_block_window,
        //                                ABlockTensor& a_scale_tile,
        //                                BBlockTensor& b_scale_tile)
        // {
        //     static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
        //                   "The CDataType as defined in traits should be the same as correspoinding "
        //                   "C block tensor data type!");

        //      auto b_scale_value = b_scale_tile.get_thread_buffer()[0];
        //     //  auto scale_reg = a_scale_tile.get_thread_buffer()[0];
        //     //  auto combined_scale = scale_reg * b_scale_value;

        //     // hot loop:
        //     static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        //         static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
        //             // read A warp tensor from A block tensor
        //             AWarpTensor a_warp_tensor;

        //             a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
        //                 merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
        //                 merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        //             static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
        //                 // read B warp tensor from B block tensor
        //                 BWarpTensor b_warp_tensor;

        //                 b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
        //                     merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
        //                     merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        //                 // read C warp tensor from C block tensor
        //                 CWarpTensor c_warp_tensor;

        //                 c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
        //                     merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
        //                     merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    
        //                 // warp GEMM
        //                 WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

        //                 c_block_tensor.set_y_sliced_thread_data(
        //                     merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
        //                     merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
        //                     c_warp_tensor.get_thread_buffer());
        //             });

        //         });
        //     });

        //     static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
        //         static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
        //         // Get the C warp tensor for this iteration
        //             CWarpTensor c_warp_tensor;
        //             c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
        //                 merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
        //                 merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        //             // Calculate scale offset
        //             index_t warp_id = __builtin_amdgcn_readfirstlane(__builtin_amdgcn_get_lane_id() / 64);
        //             constexpr index_t actual_m_pos = warp_id * 32 + mIter * 64;
        //             constexpr index_t src_reg_offset = actual_m_pos;

        //             // Apply scale to each row of C warp tensor
        //             static_for<0, WarpGemm::kM, 16>{}([&](auto c_row) {
        //                 constexpr uint32_t reg_offset_for_row_data = c_row / 16;

        //                 auto& scale_reg = a_scale_tile.get_thread_buffer()[src_reg_offset];
        //                 auto combined_scale = scale_reg * b_scale_value;

        //                 c_warp_tensor.get_thread_buffer()[reg_offset_for_row_data] *= combined_scale;
        //             });

        //             // Write back the scaled C warp tensor
        //             c_block_tensor.set_y_sliced_thread_data(
        //                 merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
        //                 merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
        //                 c_warp_tensor.get_thread_buffer());
        //         });  
        //     });  
           
        // }

        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    
                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });

                });
            });   
        }

        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow, typename CScaleBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BSmemBlockWindow& b_block_window,
                                       CScaleBlockTensor& c_scale_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;
 
                    // 从整个完整的Tensor中提取Y维度的一个切片 整个中的虚幻起始 sequence<mIter, kIter> a_warp_y_index_zeros当前这个循环的warp索引
                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros), // <mIter,kIter,mWarp,nWarps>
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));    // <1,1,m32,k32>

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros), // <
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
                        CWarpTensor c_scale_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        c_scale_warp_tensor.get_thread_buffer() = c_scale_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    
                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        
                        static_for<0, c_warp_tensor.get_thread_buffer_size(), 1>{}([&](auto i) {
                            c_warp_tensor.get_thread_buffer()[i] *= c_scale_warp_tensor.get_thread_buffer()[i];
                        });
                        
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });

                });
            });   
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
    {
        static constexpr index_t KPerThread     = GemmTraits::KPerThread;
        static constexpr index_t NumMacClusters = GemmTraits::InterWaveSchedulingMacClusters;
        static constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;
        static constexpr index_t KInnerLoopIter = KPerInnerLoop / WarpGemm::kKPerThread;

        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        ALdsTile b_warp_tile_;

        template <index_t KIdx, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(MakeBBlockDistributionEncode());

            auto a_lds_gemm_window = make_tile_window(
                a_block_window.get_bottom_tensor_view(),
                make_tuple(number<GemmTraits::MPerBlock>{}, number<KPerInnerLoop>{}),
                {0, KIdx * KPerInnerLoop},
                a_lds_load_tile_distr);
            auto b_lds_gemm_window = make_tile_window(
                b_block_window.get_bottom_tensor_view(),
                make_tuple(number<GemmTraits::NPerBlock>{}, number<KPerInnerLoop>{}),
                {0, KIdx * KPerInnerLoop},
                b_lds_load_tile_distr);

            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_lds_gemm_window);
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                load_interleaved_pk_type(b_warp_tile_, b_block_window);
            }
            else
            {
                load_tile(b_warp_tile_, b_lds_gemm_window);
            }
        }

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BSmemBlockWindow& b_block_window)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KRepeat, 1>{}([&](auto kIter) {
                LocalPrefetch<kIter.value>(a_block_window, b_block_window);
                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC
                // cluster, but except the first, as we can shorten non-MAC cluster a bit
                // and there's no observable negative impact. The desired effect is waves in
                // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                // hijacking MAC resource from other workgroups and reducing the chance of
                // latency hiding by waiting for the rest of the workgroup at the eventual
                // sync point.
                if constexpr(kIter.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }

                static_for<0, KInnerLoopIter, 1>{}([&](auto kInnerIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        // read A warp tensor from A block tensor
                        AWarpTensor a_warp_tensor;

                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kInnerIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read B warp tensor from B block tensor
                            BWarpTensor b_warp_tensor;

                            b_warp_tensor.get_thread_buffer() =
                                b_warp_tile_.get_y_sliced_thread_data(
                                    merge_sequences(sequence<nIter, kInnerIter>{},
                                                    b_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                            // read C warp tensor from C block tensor-
                            CWarpTensor c_warp_tensor;

                            c_warp_tensor.get_thread_buffer() =
                                c_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                            // The block_sync_lds() here performs double duty:
                            // A) safeguard against data hazard because barrier from
                            // blockwise_gemm is moved here B) reduce VMEM FIFO congestion
                            // by applying small delays to different wavefronts It is
                            // performed near the end of MAC cluster to minimize lgkmcnt
                            // penalty
                            if constexpr(kIter.value == KRepeat - 1 &&
                                         kInnerIter.value == KInnerLoopIter - 1 &&
                                         mIter.value == MIterPerWarp - 1 &&
                                         nIter.value == NIterPerWarp - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            // warp GEMM
                            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                            // write C warp tensor into C block tensor
                            c_block_tensor.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());

                            if constexpr(kInnerIter.value == 0 && mIter.value == 0 &&
                                         nIter.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };

    public:
    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
    }

    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_.LocalPrefetch(a_block_window, b_block_window);
    }

    // C += A * B
    // template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow, typename ABlockTensor, typename BBlockTensor>
    // CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
    //                                const ASmemBlockWindow& a_block_window,
    //                                const BSmemBlockWindow& b_block_window,
    //                                const ABlockTensor& a_scale_block_tensor,
    //                                const BBlockTensor& b_scale_block_tensor)
    // {
    //     block_gemm_impl_(c_block_tensor, a_block_window, b_block_window, a_scale_block_tensor, b_scale_block_tensor);
    // }

    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window,
                                   const CBlockTensor& c_scale_block_tensor)
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window, c_scale_block_tensor);
    }

    // C += A * B
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window);
    }

    // C = A * B
    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE auto operator()(const ASmemBlockWindow& a_block_window,
                                   const BSmemBlockWindow& b_block_window)
    {
        auto c_block_tensor = MakeCBlockTile();
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_window);
        return c_block_tensor;
    }

    private:
    BlockGemmImpl<Scheduler, Traits> block_gemm_impl_{};
};

} // namespace ck_tile
