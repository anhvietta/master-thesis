//std
#include <iostream>
#include <algorithm>
//gttl
#include <sequences/literate_multiseq.hpp>
#include <sequences/gttl_multiseq.hpp>
#include <alignment/blosum62.hpp>
#include <utilities/bitpacker.hpp>
#include <utilities/mathsupport.hpp>
#include <utilities/cxxopts.hpp>
#include <utilities/ska_lsb_radix_sort.hpp>
#include <utilities/bytes_unit.hpp>
#include <utilities/constexpr_for.hpp>
#include <utilities/constexpr_if.hpp>
#include "utilities/runtime_class.hpp"
//faiss
#include <faiss/index_io.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
//torch
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/autocast_mode.h>
#include <torch/cuda.h>
//cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

static constexpr const Blosum62 sc{};
static constexpr const auto char_spec = sc.character_spec;
static constexpr const auto undefined_rank = sc.num_of_chars;
static constexpr const auto score_matrix = sc.score_matrix;
static constexpr const faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT;
static constexpr const size_t max_length = 512;

const size_t device_idx=1;
torch::Device device(torch::kCUDA,device_idx);

static void usage(const cxxopts::Options &options)
{
    std::cerr << options.help() << std::endl;
}

class SearchOptions{
private:
    std::vector<std::string> inputfiles{};

    bool help_option = false,
    show = false,
    short_header = false,
    gpu = false,
    ivf = false,
    pq = false;

    std::string
    model_path = "./ckpts_colbert+direct_conv+relpe+match+gttl_1m_48_7_5_30.pt",
    export_index = "", import_index = "",//Index setup
    ndim = "320", nlist = "20000", subQuantizers = "40", bitsPerCode = "8", //Index options
    train_nsamples = "10000", nprobe = "400",
    target_batch_size = "1000", query_batch_size = "50", //Torch related
    k = "10", threshold = "0.4", // AA search options
    tolerance = "2", fraglen_bound = "3", //Assembly options
    gap_open = "10", gap_ext = "1", mismatch = "0" // Scoring options
    ;

public:
    SearchOptions() {};

    void parse(int argc, char **argv)
    {
        cxxopts::Options options(argv[0],"run MMseqs2 on a query & a target database");
        options.set_width(80);
        options.custom_help(std::string("[options] target_fasta query_fasta"));
        options.set_tab_expansion();
        options.add_options()
        ("e,short_header", "show header up to and excluding the first blank",
         cxxopts::value<bool>(short_header)->default_value("false"))
        ("m,model", "path to exported model",
         cxxopts::value<std::string>(model_path)->default_value("./ckpts_colbert+direct_conv+relpe+match+gttl_1m_48_7_5_30.pt"))
        ("g,gpu", "use GPU",
         cxxopts::value<bool>(gpu)->default_value("false"))
        ("ivf", "use IVF type index",
         cxxopts::value<bool>(ivf)->default_value("false"))
        ("pq", "use PQ type index",
         cxxopts::value<bool>(pq)->default_value("false"))
        ("ndim", "number of vector dimensions output by the model",
         cxxopts::value<std::string>(ndim)->default_value("320"))
        ("nlist", "IVF Options: number of k-means clusters",
         cxxopts::value<std::string>(nlist)->default_value("20000"))
        ("nprobe", "IVF Options: number of Voronoi cells to probe",
         cxxopts::value<std::string>(nprobe)->default_value("800"))
        ("subquantizers", "PQ Options: number of subquantizers",
         cxxopts::value<std::string>(subQuantizers)->default_value("40"))
        ("bitspercode", "PQ Options: bits per encoding dimension",
         cxxopts::value<std::string>(bitsPerCode)->default_value("8"))
        ("train_samples", "IVF/PQ: Number of training sequences",
         cxxopts::value<std::string>(train_nsamples)->default_value("10000"))
        ("k,knn", "set k-NN",
         cxxopts::value<std::string>(k)->default_value("10"))
        ("export_index", "export index",
         cxxopts::value<std::string>(export_index)->default_value(""))
        ("t,threshold", "set threshold",
         cxxopts::value<std::string>(threshold)->default_value("0.4"))
        ("l,tolerance", "set tolerance",
         cxxopts::value<std::string>(tolerance)->default_value("2"))
        ("f,fraglen_bound", "set fragment length lower bound",
         cxxopts::value<std::string>(fraglen_bound)->default_value("3"))
        ("gap_open", "set gap open",
         cxxopts::value<std::string>(gap_open)->default_value("10"))
        ("gap_ext", "set gap extend",
         cxxopts::value<std::string>(gap_ext)->default_value("1"))
        ("mismatch", "set mismatch",
         cxxopts::value<std::string>(mismatch)->default_value("0"))
        ("b,target_batch_size", "set batch size when creating index",
         cxxopts::value<std::string>(target_batch_size)->default_value("1000"))
        ("q,query_batch_size", "set batch size when query",
         cxxopts::value<std::string>(query_batch_size)->default_value("50"))
        ("i,index", "input index",
         cxxopts::value<std::string>(import_index)->default_value(""))
        ("s,show", "output matches",
         cxxopts::value<bool>(show)->default_value("false"))
        ("h,help", "print usage");
        try
        {
            auto result = options.parse(argc, argv);
            if (result.count("help") > 0)
            {
                help_option = true;
                usage(options);
            } else
            {
                const std::vector<std::string>& unmatched_args = result.unmatched();
                if(unmatched_args.size() != 2){
                    throw cxxopts::OptionException("Exact 2 fasta files are needed.");
                }
                for (size_t idx = 0; idx < unmatched_args.size(); idx++)
                {
                    inputfiles.push_back(unmatched_args[idx]);
                }
            }
        }
        catch (const cxxopts::OptionException &e)
        {
            usage(options);
            throw std::invalid_argument(e.what());
        }
    }
    bool help_option_is_set(void) const noexcept
    {
        return help_option;
    }
    bool short_header_option_is_set(void) const noexcept
    {
        return short_header;
    }
    bool show_option_is_set(void) const noexcept
    {
        return show;
    }
    bool gpu_option_is_set(void) const noexcept
    {
        return gpu;
    }
    bool ivf_option_is_set(void) const noexcept
    {
        return ivf;
    }
    bool pq_option_is_set(void) const noexcept
    {
        return pq;
    }
    const std::string &model_path_get(void) const noexcept
    {
        return model_path;
    }
    const std::string &import_index_get(void) const noexcept
    {
        return import_index;
    }
    const std::string &export_index_get(void) const noexcept
    {
        return export_index;
    }
    const std::string &ndim_get(void) const noexcept
    {
        return ndim;
    }
    const std::string &nlist_get(void) const noexcept
    {
        return nlist;
    }
    const std::string &nprobe_get(void) const noexcept
    {
        return nprobe;
    }
    const std::string &subquantizers_get(void) const noexcept
    {
        return subQuantizers;
    }
    const std::string &bitsPerCode_get(void) const noexcept
    {
        return bitsPerCode;
    }
    const std::string &train_nsamples_get(void) const noexcept
    {
        return train_nsamples;
    }
    const std::string &knn_get(void) const noexcept
    {
        return k;
    }
    const std::string &threshold_get(void) const noexcept
    {
        return threshold;
    }
    const std::string &tolerance_get(void) const noexcept
    {
        return tolerance;
    }
    const std::string &fraglen_bound_get(void) const noexcept
    {
        return fraglen_bound;
    }
    const std::string &gap_open_get(void) const noexcept
    {
        return gap_open;
    }
    const std::string &gap_ext_get(void) const noexcept
    {
        return gap_ext;
    }
    const std::string &mismatch_get(void) const noexcept
    {
        return mismatch;
    }
    const std::string &target_batch_size_get(void) const noexcept
    {
        return target_batch_size;
    }
    const std::string &query_batch_size_get(void) const noexcept
    {
        return query_batch_size;
    }
    const std::vector<std::string> &inputfiles_get(void) const noexcept
    {
        return inputfiles;
    }
};

using Sample = torch::data::Example<torch::Tensor, torch::Tensor>;

class SequenceDataset : public torch::data::datasets::Dataset<SequenceDataset, Sample> {
private:
    GttlMultiseq* multiseq;
    size_t dsize = 0;
public:
    SequenceDataset(GttlMultiseq* mseq){
        multiseq = mseq;
        dsize = multiseq->sequences_number_get();
    }

    // Equivalent of __len__
    std::optional<size_t> size() const override {
        return dsize;
    }

    // Equivalent of __getitem__
    Sample get(size_t idx) override {
        const auto seqlen = multiseq->sequence_length_get(idx);
        auto sequence_ptr = multiseq->sequence_ptr_get(idx);
        std::vector<uint8_t> encoded(max_length, undefined_rank-1);
        for (size_t i = 0; i < seqlen; i++) {
            encoded[i] = sequence_ptr[i];
        }
        torch::Tensor sequence = torch::from_blob(encoded.data(), max_length, torch::kUInt8).clone();
        auto len = torch::from_blob((uint16_t*)&seqlen, {1}, torch::kUInt16).clone();
        return {sequence, len};
    }

    void set_size(const size_t new_size){
        dsize = std::min(new_size, dsize);
    }

    /*std::tuple<torch::Tensor, torch::Tensor> get(size_t idx) {
     *        const auto seqlen = multiseq->sequence_length_get(idx);
     *        auto sequence_ptr = multiseq->sequence_ptr_get(idx);
     *        std::vector<uint8_t> encoded(max_length, undefined_rank-1);
     *        for (size_t i = 0; i < seqlen; i++) {
     *            encoded[i] = sequence_ptr[i];
}
torch::Tensor sequence = torch::from_blob(encoded.data(), max_length, torch::kUInt8).clone();
return std::make_tuple(sequence, torch::from_blob((uint16_t*)&seqlen, {1}, torch::kUInt16).clone());
}*/

    std::shared_ptr<SequenceDataset> clone() const {
        return std::make_shared<SequenceDataset>(*this);
    }
};

std::tuple<std::vector<c10::IValue>,std::vector<uint16_t>> make_input(const std::vector<Sample>& samples){
    const size_t batch_size = samples.size();
    std::vector<torch::Tensor> seq{};
    seq.reserve(batch_size);
    std::vector<uint16_t> seqlen{};
    seqlen.reserve(batch_size);

    for(const auto& sample : samples){
        seq.push_back(sample.data.to(device).to(torch::kFloat32));
        seqlen.push_back(static_cast<uint16_t>(sample.target.item<int64_t>()));
    }

    const auto seq_stacked = torch::stack(seq);
    std::vector<c10::IValue> stacked_vec{};
    stacked_vec.push_back(c10::IValue(seq_stacked));

    return std::make_tuple(stacked_vec, seqlen);
}

torch::jit::script::Module get_model(const std::string& ckpts){
    torch::jit::script::Module module = torch::jit::load(ckpts);
    module.to(device);
    return module;
}

class HitIndex {
public:
    uint64_t seq_idx=0;
    uint16_t pos_idx=0;

    HitIndex(){};
    HitIndex(const uint64_t _seq_idx, const uint16_t _pos_idx){
        seq_idx = _seq_idx;
        pos_idx = _pos_idx;
    }
};

void make_doc_ids(
    const std::vector<uint16_t>& seqlen,
    size_t base_doc_id,
    std::vector<HitIndex>& ids) {
    for (uint64_t j = 0; j < seqlen.size(); ++j) {
        for (uint16_t k = 0; k < seqlen[j]; ++k) {
            ids.emplace_back(base_doc_id + j, k);
        }
    }
}

std::vector<HitIndex> reconstruct_id(
    GttlMultiseq* multiseq
){
    std::vector<HitIndex> ids{};
    for(size_t i = 0; i < multiseq->sequences_number_get(); i++){
        for(size_t j = 0; j < multiseq->sequence_length_get(i); j++){
            ids.emplace_back(i, j);
        }
    }
    return ids;
}

template<bool gpu>
class FlatIndex {
    using IndexType = std::conditional_t<gpu,
    faiss::Index,
    faiss::IndexFlatIP
    >;
    using ConfigType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlatConfig,
    size_t
    >;
    using ExportType = faiss::Index;
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
    ConfigType* cfg;
    size_t ndim;
public:
    IndexType* index;

    FlatIndex(const size_t ndim_){
        ndim = ndim_;
    }

    void init(){
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            cfg = new ConfigType();
            cfg->useFloat16 = false;
            cfg->device = device_idx;
            index = new IndexType(res.get(), ndim, *cfg);
        } else {
            index = new IndexType(ndim);
        }
    }

    void export_index(const std::string& export_path) const {
        if constexpr(gpu){
            ExportType* e_index = faiss::gpu::index_gpu_to_cpu(index);
            faiss::write_index(e_index, export_path.c_str());
            delete e_index;
        } else {
            faiss::write_index(index, export_path.c_str());
        }
    }

    void import_index(const std::string& import_path) {
        ExportType* e_index = faiss::read_index(import_path.c_str());
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            index = faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, e_index);
        } else {
            index = dynamic_cast<IndexType*>(e_index);
        }
        //delete e_index;
    }

    void set_pq_params(const size_t subQuantizers, const size_t bitsPerCode) const {}

    void train(const size_t n, const float* x){}

    ~FlatIndex(){
        if(index){
            delete index;
        }
        if(cfg){
            delete cfg;
        }
    }
};

template<bool gpu>
class IVFIndex {
    using IndexType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexIVFFlat,
    faiss::IndexIVFFlat
    >;
    using ConfigType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexIVFFlatConfig,
    size_t
    >;
    using QuantizerType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlat,
    faiss::IndexFlat
    >;
    using QuantizerConfigType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlatConfig,
    size_t
    >;
    using ExportType = faiss::Index;

    QuantizerType* quantizer;
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
    ConfigType* cfg;
    QuantizerConfigType* quantizerCfg;
    size_t ndim;
    size_t nlist;

public:
    IndexType* index;
    IVFIndex(const size_t ndim_){
        ndim = ndim_;
    }

    void set_nlist(const size_t nlist_){
        nlist = nlist_;
    }

    void set_nprobe(const size_t nprobe_){
        index->nprobe = nprobe_;
    }

    void init(){
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            cfg = new ConfigType();
            //cfg->useFloat16 = false;
            cfg->device = device_idx;
            quantizerCfg = new QuantizerConfigType();
            quantizerCfg->useFloat16 = false;
            quantizerCfg->device = device_idx;

            quantizer = new QuantizerType(res.get(), ndim, metric, *quantizerCfg);
            index = new IndexType(res.get(), quantizer, ndim, nlist, metric, *cfg);
        } else {
            quantizer = new QuantizerType(ndim);
            index = new IndexType(quantizer, ndim, nlist, metric);
        }
    }

    void export_index(const std::string& export_path) const {
        if constexpr(gpu){
            ExportType* e_index = faiss::gpu::index_gpu_to_cpu(index);
            faiss::write_index(e_index, export_path.c_str());
            delete e_index;
        } else {
            faiss::write_index(index, export_path.c_str());
        }
    }

    void import_index(const std::string& import_path) {
        ExportType* e_index = faiss::read_index(import_path.c_str());
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            index = dynamic_cast<IndexType*>(faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, e_index));
        } else {
            index = dynamic_cast<IndexType*>(e_index);
        }
        //delete e_index;
    }

    void set_pq_params(const size_t subQuantizers, const size_t bitsPerCode) const {}
    void train_index(const size_t n, const float* x){
        index->train(n,x);
    }

    ~IVFIndex(){
        if(quantizer){
            delete quantizer;
        }
        if(index){
            delete index;
        }
        if(cfg){
            delete cfg;
        }
        if(quantizerCfg){
            delete quantizerCfg;
        }
    }
};

template<bool gpu>
class PQIndex {
    using IndexType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexIVFPQ,
    faiss::IndexIVFPQ
    >;
    using ConfigType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexIVFPQConfig,
    size_t
    >;
    using QuantizerType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlat,
    faiss::IndexFlat
    >;
    using QuantizerConfigType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlatConfig,
    size_t
    >;
    using ExportType = faiss::Index;

    QuantizerType* quantizer;
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
    ConfigType* cfg;
    QuantizerConfigType* quantizerCfg;
    size_t ndim;
    size_t nlist;
    size_t subQuantizers;
    size_t bitsPerCode;

public:
    IndexType* index;

    PQIndex(const size_t ndim_){
        ndim = ndim_;
    }

    void set_nlist(const size_t nlist_){
        nlist = nlist_;
    }

    void set_nprobe(const size_t nprobe_){
        index->nprobe = nprobe_;
    }

    void set_pq_params(const size_t subQuantizers_, const size_t bitsPerCode_){
        subQuantizers = subQuantizers_;
        bitsPerCode = bitsPerCode_;
    }

    void init(){
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            cfg = new ConfigType();
            //cfg->useFloat16 = false;
            cfg->device = device_idx;
            quantizerCfg = new QuantizerConfigType();
            quantizerCfg->useFloat16 = false;
            quantizerCfg->device = device_idx;
            quantizer = new QuantizerType(res.get(), ndim, metric, *quantizerCfg);
            index = new IndexType(res.get(), quantizer, ndim, nlist, subQuantizers, bitsPerCode, metric, *cfg);
        } else {
            quantizer = new QuantizerType(ndim);
            index = new IndexType(quantizer, ndim, nlist, subQuantizers, bitsPerCode, metric);
        }
    }

    void export_index(const std::string& export_path) const {
        if constexpr(gpu){
            ExportType* e_index = faiss::gpu::index_gpu_to_cpu(index);
            faiss::write_index(e_index, export_path.c_str());
            delete e_index;
        } else {
            faiss::write_index(index, export_path.c_str());
        }
    }

    void import_index(const std::string& import_path) {
        ExportType* e_index = faiss::read_index(import_path.c_str());
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            index = dynamic_cast<IndexType*>(faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, e_index));
        } else {
            index = dynamic_cast<IndexType*>(e_index);
        }
    }

    void train_index(const size_t n, const float* x){
        index->train(n,x);
    }

    ~PQIndex(){
        if(quantizer){
            delete quantizer;
        }
        if(index){
            delete index;
        }
        if(cfg){
            delete cfg;
        }
        if(quantizerCfg){
            delete quantizerCfg;
        }
    }
};

template<bool gpu, bool ivf, bool pq>
struct IndexClassSelector;

template<bool gpu> struct IndexClassSelector<gpu, true,  true>  { using type = PQIndex<gpu>; };
template<bool gpu> struct IndexClassSelector<gpu, true,  false> { using type = IVFIndex<gpu>; };
template<bool gpu> struct IndexClassSelector<gpu, false, true>  { using type = FlatIndex<gpu>; };
template<bool gpu> struct IndexClassSelector<gpu, false, false>  { using type = FlatIndex<gpu>; };

template<bool gpu, bool ivf, bool pq>
void train_index(const torch::jit::script::Module& model,
                    GttlMultiseq* multiseq,
                    const uint64_t batch_size,
                    typename IndexClassSelector<gpu,ivf,pq>::type* index,
                    size_t nsamples,
                    const size_t ndim
){
    auto literate_multiseq = new LiterateMultiseq<char_spec,undefined_rank>(multiseq);
    literate_multiseq->perform_sequence_encoding();
    SequenceDataset dataset{multiseq};
    dataset.set_size(nsamples);
    nsamples = dataset.size().value_or(0);
    auto options = torch::data::DataLoaderOptions();
    options.batch_size(batch_size);
    const auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset,
        options
    );

    torch::NoGradGuard no_grad;
    std::vector<float> v_accumulator{};
    v_accumulator.reserve(max_length * nsamples * ndim);
    for (auto& batch : *dataloader){
        //std::cout << "Getting batch" << std::endl;
        auto [seq, seqlen] = make_input(batch);
        torch::jit::Method encode = model.get_method("encode");
        auto enc = encode(seq).toTensor();
        // Ensure CPU float tensor
        enc = enc.to(torch::kCPU).to(torch::kFloat32);

        const auto B = enc.size(0);
        //const auto L = enc.size(1);
        const auto D = enc.size(2);

        //std::cout << "Parsing encodings" << std::endl;
        // Compute total number of valid positions to reserve memory
        int64_t total_rows = 0;
        for (int64_t b = 0; b < B; ++b) {
            total_rows += seqlen[b];
        }

        auto seqacc = enc.accessor<float,3>(); // B x L x D

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t l = 0; l < seqlen[b]; ++l) {
                for (int64_t d = 0; d < D; ++d) {
                    v_accumulator.push_back(seqacc[b][l][d]);
                }
            }
        }
    }
    index->index->train(nsamples, v_accumulator.data());
    delete literate_multiseq;
}

template<bool gpu, bool ivf, bool pq>
std::vector<HitIndex> build_index(
    const torch::jit::script::Module& model,
    GttlMultiseq* multiseq,
    const uint64_t batch_size,
    typename IndexClassSelector<gpu, ivf, pq>::type* index
) {
    //std::cout << "Creating dataset" << std::endl;
    auto literate_multiseq = new LiterateMultiseq<char_spec,undefined_rank>(multiseq);
    literate_multiseq->perform_sequence_encoding();
    SequenceDataset dataset{multiseq};

    //std::cout << "Creating dataloader" << std::endl;
    auto options = torch::data::DataLoaderOptions();
    options.batch_size(batch_size);
    const auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        //torch::data::datasets::BatchDataset<SequenceDataset>(dataset),
        dataset,
        options
    );
    //std::cout << "Created" << std::endl;

    torch::NoGradGuard no_grad;
    std::vector<HitIndex> ids{};
    size_t base_doc_id = 0;
    for (auto& batch : *dataloader){
        //std::cout << "Getting batch" << std::endl;
        auto [seq, seqlen] = make_input(batch);
        torch::jit::Method encode = model.get_method("encode");
        auto enc = encode(seq).toTensor();
        // Ensure CPU float tensor
        enc = enc.to(torch::kCPU).to(torch::kFloat32);

        const auto B = enc.size(0);
        //const auto L = enc.size(1);
        const auto D = enc.size(2);

        //std::cout << "Parsing encodings" << std::endl;
        // Compute total number of valid positions to reserve memory
        int64_t total_rows = 0;
        for (int64_t b = 0; b < B; ++b) {
            total_rows += seqlen[b];
        }
        std::vector<float> valid_enc_flat;
        valid_enc_flat.reserve(total_rows * D); // each row has D features

        auto seqacc = enc.accessor<float,3>(); // B x L x D

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t l = 0; l < seqlen[b]; ++l) {
                for (int64_t d = 0; d < D; ++d) {
                    valid_enc_flat.push_back(seqacc[b][l][d]);
                }
            }
        }

        // Add to FAISS index
        index->index->add(total_rows, valid_enc_flat.data());
        make_doc_ids(seqlen, base_doc_id, ids);
        base_doc_id += batch_size;
    }
    delete literate_multiseq;
    return ids;
}

std::vector<std::tuple<size_t, size_t, float>> add_triplets(
    std::vector<std::tuple<size_t, size_t, float>>& triplets,
    const float* D,     // distances, shape (L*K)
    const faiss::idx_t* I,   // indices, shape (L*K)
    size_t L,
    size_t K,
    float t,
    size_t offset)
{
    for (size_t x = 0; x < L; ++x) {
        size_t row_offset = x * K;
        for (size_t k = 0; k < K; ++k) {
            float d = D[row_offset + k];
            if (d >= t) {
                const auto y = static_cast<size_t>(I[row_offset + k]);
                triplets.emplace_back(
                    static_cast<size_t>(x)+offset,y,d
                );
            }
        }
    }
    return triplets;
}

template<bool gpu, bool ivf, bool pq>
std::tuple<std::vector<std::tuple<size_t, size_t, float>>, std::vector<HitIndex>> kmer_query(
    const torch::jit::script::Module& model,
    GttlMultiseq* multiseq,
    const uint64_t batch_size,
    //faiss::Index* index,
    typename IndexClassSelector<gpu,ivf,pq>::type* index,
    const uint64_t k,
    const float threshold
) {
    auto literate_multiseq = new LiterateMultiseq<char_spec,undefined_rank>(multiseq);
    literate_multiseq->perform_sequence_encoding();
    const SequenceDataset dataset{multiseq};
    auto options = torch::data::DataLoaderOptions();
    options.batch_size(batch_size);
    const auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        //torch::data::datasets::BatchDataset<SequenceDataset>(dataset),
        dataset,
        options
    );
    torch::NoGradGuard no_grad;
    std::vector<HitIndex> ids{};
    size_t base_doc_id = 0;
    size_t batch_offset = 0;
    std::vector<std::tuple<uint64_t, uint64_t, float>> triplets{};
    
    std::chrono::duration<double> encode_duration{0};
    std::chrono::duration<double> query_duration{0};
    
    for (auto& batch : *dataloader){
	const auto start = std::chrono::high_resolution_clock::now();
        auto [seq, seqlen] = make_input(batch);
        torch::jit::Method encode = model.get_method("encode");
        auto enc = encode(seq).toTensor();
        // Ensure CPU float tensor
        enc = enc.to(torch::kCPU).to(torch::kFloat32);
	const auto time_encode = std::chrono::high_resolution_clock::now();

        const auto B = enc.size(0);
        //const auto L = enc.size(1);
        const auto D = enc.size(2);

        // Compute total number of valid positions to reserve memory
        int64_t total_rows = 0;
        for (int64_t b = 0; b < B; ++b) {
            total_rows += seqlen[b];
        }
        std::vector<float> valid_enc_flat;
        valid_enc_flat.reserve(total_rows * D); // each row has D features
        auto distances = new float[total_rows * k];
        auto labels = new faiss::idx_t[total_rows * k];

        auto seqacc = enc.accessor<float,3>(); // B x L x D

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t l = 0; l < seqlen[b]; ++l) {
                for (int64_t d = 0; d < D; ++d) {
                    valid_enc_flat.push_back(seqacc[b][l][d]);
                }
            }
        }

        index->index->search(total_rows, valid_enc_flat.data(), k, distances, labels);
	const auto time_query = std::chrono::high_resolution_clock::now();
        make_doc_ids(seqlen, base_doc_id, ids);
        add_triplets(triplets, distances, labels, total_rows, k, threshold, batch_offset);
        base_doc_id += batch_size;
	batch_offset += total_rows;
        delete distances;
        delete labels;

	encode_duration += time_encode - start;
	query_duration += time_query - time_encode;
    }
    std::cout << "#Time:\tEncode (ms):\t" << encode_duration.count() * 1000 << std::endl;
    std::cout << "#Time:\tQuery (ms):\t" << query_duration.count() * 1000 << std::endl;

    delete literate_multiseq;
    return std::make_tuple( triplets, ids );
}

/*struct Hit {
    *    size_t target_seq_idx;
    *    size_t query_seq_idx;
    *    size_t target_pos_idx;
    *    size_t query_pos_idx;
    *    float score;
    *
    *    Hit(const size_t tsi,const size_t qsi,const size_t tpi,const size_t qpi,const float d){
    *        target_seq_idx = tsi;
    *        query_seq_idx = qsi;
    *        target_pos_idx = tpi;
    *        query_pos_idx = qpi;
    *        score = d;
    *    }
    * };*/

uint8_t sizeof_unit_get(const size_t total_bits) {
    if(total_bits <= 64) return 8;
    if(total_bits % 8 == 0) return total_bits/8;
    return total_bits/8+1;
}


template<size_t sizeof_hit_unit>
std::vector<BytesUnit<sizeof_hit_unit,4>> triplet_to_hit(
    const std::vector<std::tuple<size_t, size_t, float>>& triplets,
    const std::vector<HitIndex>& query_ids,
    const std::vector<HitIndex>& target_ids,
    const GttlBitPacker<sizeof_hit_unit,4>& hit_packer,
    const size_t query_max_len
) {
    std::vector<BytesUnit<sizeof_hit_unit,4>> hits{};
    hits.reserve(triplets.size());
    for(const auto& triplet : triplets){
        const auto qi = std::get<0>(triplet);
        const auto ti = std::get<1>(triplet);
        //const auto d = std::get<2>(triplet);
        const auto query_hi = query_ids[qi];
        const auto target_hi = target_ids[ti];
        const int64_t diagonal =  static_cast<int64_t>(target_hi.pos_idx)-static_cast<int64_t>(query_hi.pos_idx)+query_max_len;
        hits.emplace_back(hit_packer, std::array<uint64_t,4>{
            static_cast<uint64_t>(target_hi.seq_idx),
                            static_cast<uint64_t>(query_hi.seq_idx),
                            static_cast<uint64_t>(diagonal),
                            static_cast<uint64_t>(query_hi.pos_idx)
        });
    }
    return hits;
}

template<size_t sizeof_match_unit>
void hit_sort(std::vector<BytesUnit<sizeof_match_unit,4>>& container, const size_t sort_bits) {
    if constexpr (sizeof_match_unit == 8){
        ska_lsb_radix_sort<uint64_t>(sort_bits,
                                        reinterpret_cast<uint64_t *>
                                        (container.data()),
                                        container.size());
    } else {
        ska_large_lsb_small_radix_sort(sizeof_match_unit,sort_bits,reinterpret_cast<uint8_t *>(
            container.data()),container.size(),
                                        false
        );
    }
}

bool is_index(const std::string& inputfname){
    const std::string ending = ".faiss";
    if (inputfname.length() >= ending.length()) {
        return (0 == inputfname.compare (inputfname.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

template<size_t sizeof_diag_unit>
std::tuple<std::vector<std::tuple<size_t,size_t,size_t,size_t>>,std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>>> create_groups(
    const std::vector<BytesUnit<sizeof_diag_unit,5>>& diag_vec,
    const GttlBitPacker<sizeof_diag_unit,5>& diag_packer,
    const size_t query_max_len
) {
    assert(diag_vec.size() > 0);
    std::vector<std::tuple<size_t,size_t,size_t,size_t>> indexgroup_vec;
    std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>> diaggroup_vec;
    size_t curr_target_seqnum = diag_vec[0].template decode_at<0>(diag_packer);
    size_t curr_query_seqnum = diag_vec[0].template decode_at<1>(diag_packer);
    size_t diagnum = diag_vec[0].template decode_at<2>(diag_packer);
    size_t start_on_query = diag_vec[0].template decode_at<3>(diag_packer);
    size_t end_on_query = diag_vec[0].template decode_at<4>(diag_packer);
    diaggroup_vec.emplace_back(
        diagnum + start_on_query - query_max_len,
        start_on_query,
        diagnum + end_on_query - query_max_len,
        end_on_query
    );
    size_t start_index = 0;
    for(size_t i = 1; i < diag_vec.size(); i++){
        const auto diag = diag_vec[i];
        const auto target_seqnum = diag.template decode_at<0>(diag_packer);
        const auto query_seqnum = diag.template decode_at<1>(diag_packer);
        diagnum = diag.template decode_at<2>(diag_packer);
        start_on_query = diag.template decode_at<3>(diag_packer);
        end_on_query = diag.template decode_at<4>(diag_packer);
        if(target_seqnum != curr_target_seqnum || query_seqnum != curr_query_seqnum){
            indexgroup_vec.emplace_back(
                curr_target_seqnum,
                curr_query_seqnum,
                start_index,
                i
            );
            curr_target_seqnum = target_seqnum;
            curr_query_seqnum = query_seqnum;
            start_index = i;
        }
        diaggroup_vec.emplace_back(
            diagnum + start_on_query - query_max_len,
            start_on_query,
            diagnum + end_on_query - query_max_len,
            end_on_query
        );
    }
    indexgroup_vec.emplace_back(
        curr_target_seqnum,
        curr_query_seqnum,
        start_index,
        diag_vec.size()
    );
    return std::make_tuple(indexgroup_vec, diaggroup_vec);
}

void set_bits(std::bitset<max_length>& bitset, const size_t start, const size_t end){
    for (size_t i = start; i <= end; ++i){
        bitset.set(max_length - i - 1);
    }
}

std::tuple<std::bitset<max_length>,std::bitset<max_length>> diags_to_bitset(
    const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>& scored_diag
){
    std::bitset<max_length> target_bits{};
    std::bitset<max_length> query_bits{};
    for(const auto& diag : scored_diag){
        const auto& [st, sq, et, eq, s] = diag;
        set_bits(target_bits, st, et);
        set_bits(query_bits, sq, eq);
    }
    return std::make_tuple(target_bits, query_bits);
}

bool diag_precede(
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& A,
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& B
) {
    return std::get<2>(A) < std::get<0>(B) && std::get<3>(A) < std::get<1>(B);
}

bool diag_comp(
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& A,
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& B
){
    if(std::get<2>(A) < std::get<2>(B)){
        return true;
    }
    if(std::get<2>(A) > std::get<2>(B)){
        return false;
    }
    if(std::get<3>(A) < std::get<3>(B)){
        return true;
    }
    return false;
}

std::vector<size_t> get_preceding_match(
    const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>& scored_diag,
    size_t match_idx
){
    std::vector<size_t> valid_precede{};
    const auto& m = scored_diag[match_idx];
    for(size_t i = match_idx - 1; i < scored_diag.size(); i--){
        if(diag_precede(scored_diag[i],m)){
            valid_precede.push_back(i);
        }
    }
    return valid_precede;
}

struct LocalChainCacheElem {
    float score{};
    std::vector<size_t> mlist{};
};

float score_diag_gap(
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& precede,
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>& curr,
    const float gap_open,
    const float gap_ext,
    const float mismatch
){
    assert(diag_precede(precede, curr));
    float penalty = 0;
    const auto pet = std::get<2>(precede);
    const auto peq = std::get<3>(precede);
    const auto cst = std::get<0>(curr);
    const auto csq = std::get<1>(curr);
    if(pet < cst - 1){
        if(peq < csq - 1){
            const uint16_t ma = std::max(csq - 1 - peq, cst - 1 - pet);
            const uint16_t mi = std::min(csq - 1 - peq, cst - 1 - pet);
            penalty += mismatch * mi + gap_open + (ma - mi) * gap_ext;
        } else {
            penalty += gap_open + (std::get<0>(curr) - 1 - std::get<2>(precede)) * gap_ext;
        }
    }
    if(peq < csq - 1){
        penalty += gap_open + (std::get<1>(curr) - 1 - std::get<3>(precede)) * gap_ext;
    }
    return penalty;
}

void score_cache(
    const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>& scored_diag,
    const size_t match_idx,
    std::vector<LocalChainCacheElem>& cache,
    const float gap_open,
    const float gap_ext,
    const float mismatch
){
    const auto valid_precede = get_preceding_match(scored_diag, match_idx);
    //if(valid_precede.size() > 0){
    float score = 0;
    float curr_score = 0;
    size_t best_precede = scored_diag.size();
    for(const auto& precede_idx : valid_precede){
        const float gap_penalty = score_diag_gap(
            scored_diag[precede_idx],
            scored_diag[match_idx],
            gap_open,
            gap_ext,
            mismatch
        );
        curr_score = cache[precede_idx].score - gap_penalty;
        //std::cout << curr_score << "\t" << score << "\t" << std::get<4>(scored_diag[match_idx]) << "\t" << cache[precede_idx].score << "\t" << gap_penalty << std::endl;
        if(curr_score > score){
            score = curr_score;
            best_precede = precede_idx;
        }
    }
    cache[match_idx].score = score + std::get<4>(scored_diag[match_idx]);
    if(best_precede < scored_diag.size()){
        cache[match_idx].mlist = cache[best_precede].mlist;
    }
    cache[match_idx].mlist.push_back(match_idx);
    /*} else {
        *        cache[match_idx].score = std::get<4>(scored_diag[match_idx]);
        *        cache[match_idx].mlist.push_back(match_idx);
}*/
}


std::tuple<std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>, float> chain_diagonals(
    const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>& scored_diag,
    const float gap_open,
    const float gap_ext,
    const float mismatch
) {
    std::vector<LocalChainCacheElem> cache{scored_diag.size()};
    for(size_t i = 0; i < scored_diag.size(); i++){
        score_cache(scored_diag, i, cache, gap_open, gap_ext, mismatch);
    }
    size_t idx = 0;
    float score = 0;
    for(size_t i = 0; i < scored_diag.size(); i++){
        if(cache[i].score > score){
            score = cache[i].score;
            idx = i;
        }
    }
    std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>> diags{};
    for(const auto& i : cache[idx].mlist){
        diags.push_back(scored_diag[i]);
    }

    return std::make_tuple(diags, score);
}

class Alignment {
public:
    size_t target_seq;
    size_t query_seq;
    float score;
    std::bitset<max_length> target_bits;
    std::bitset<max_length> query_bits;
    size_t num_diag;

    Alignment(
        const size_t target_seqnum,
        const size_t query_seqnum,
        const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>>& scored_diag,
        const float gap_open,
        const float gap_ext,
        const float mismatch
    ){
        target_seq = target_seqnum;
        query_seq = query_seqnum;
        const auto [aln_diags, s] = chain_diagonals(scored_diag, gap_open, gap_ext, mismatch);
        num_diag = aln_diags.size();
        score = s;
        std::tie(target_bits, query_bits) = diags_to_bitset(aln_diags);
    }

    bool test() const {
        if(target_bits.count() != query_bits.count()){
            return false;
        }
        return true;
    }
};

bool score_comp(const Alignment& a, const Alignment& b){
    return a.query_seq < b.query_seq;
}

template<size_t sizeof_match_unit, size_t sizeof_diag_unit>
std::vector<BytesUnit<sizeof_diag_unit,5>> match_to_diag(
    const std::vector<BytesUnit<sizeof_match_unit,4>>& matches,
    const GttlBitPacker<sizeof_match_unit,4>& match_packer,
    const GttlBitPacker<sizeof_diag_unit,5>& diag_packer,
    const size_t tolerance,
    const size_t fraglen_bound
){
    std::vector<BytesUnit<sizeof_diag_unit,5>> diag_vec;
    size_t curr_target_seqnum = matches[0].template decode_at<0>(match_packer);
    size_t curr_query_seqnum = matches[0].template decode_at<1>(match_packer);
    size_t curr_diag = matches[0].template decode_at<2>(match_packer);
    size_t start_on_query = matches[0].template decode_at<3>(match_packer);
    size_t end_on_query = start_on_query;
    size_t count = 0;
    for(size_t i = 1; i < matches.size(); i++){
        const auto hit = matches[i];
        const auto target_seqnum = hit.template decode_at<0>(match_packer);
        const auto query_seqnum = hit.template decode_at<1>(match_packer);
        const auto diag = hit.template decode_at<2>(match_packer);
        const auto pos = hit.template decode_at<3>(match_packer);
        if(curr_target_seqnum == target_seqnum && curr_query_seqnum ==
            query_seqnum && curr_diag == diag && pos - end_on_query <= tolerance){
            count += 1;
        end_on_query = pos;
            } else {
                if(count >= fraglen_bound){
                    diag_vec.emplace_back(diag_packer,std::array<uint64_t,5>{
                        static_cast<uint64_t>(curr_target_seqnum),
                                            static_cast<uint64_t>(curr_query_seqnum),
                                            static_cast<uint64_t>(curr_diag),
                                            static_cast<uint64_t>(start_on_query),
                                            static_cast<uint64_t>(end_on_query)
                    });
                }
                curr_target_seqnum = target_seqnum;
                curr_query_seqnum = query_seqnum;
                curr_diag = diag;
                start_on_query = pos;
                end_on_query = pos;
                count = 0;
            }
    }
    if(count >= fraglen_bound){
        diag_vec.emplace_back(diag_packer,std::array<uint64_t,5>{
            static_cast<uint64_t>(curr_target_seqnum),
                                static_cast<uint64_t>(curr_query_seqnum),
                                static_cast<uint64_t>(curr_diag),
                                static_cast<uint64_t>(start_on_query),
                                static_cast<uint64_t>(end_on_query)
        });
    }

    return diag_vec;
}

class DiagonalCut {
    std::set<uint16_t> cut{};
public:
    std::pair<uint16_t, uint16_t> start{};
    std::pair<uint16_t, uint16_t> end{};

    DiagonalCut(
        const size_t st,
        const size_t sq,
        const size_t et,
        const size_t eq
    ){
        start.first = st;
        start.second = sq;
        end.first = et;
        end.second = eq;
    }

    void insert(const uint16_t cutcoord){
        cut.insert(cutcoord);
    }

    void merge_cut_diag(
        std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>>& diags
    ) const {
        std::vector<int16_t> cut_vec{};
        cut_vec.push_back(static_cast<int16_t>(start.first)-1);
        cut_vec.insert(std::next(cut_vec.begin(), 1), cut.begin(), cut.end());
        cut_vec.push_back(end.first);
        for(size_t i = 0; i < cut_vec.size()-1; i++){
            const auto st = cut_vec[i]+1;
            const auto et = cut_vec[i+1];
            const auto sq = start.second + (st - start.first);
            const auto eq = end.second + (et - end.first);
            diags.emplace_back(
                static_cast<uint16_t>(st),
                                static_cast<uint16_t>(sq),
                                static_cast<uint16_t>(et),
                                static_cast<uint16_t>(eq)
            );
        }
    }
};

std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>> split_diag(
    const std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>* const diag_data,
    const size_t start,
    const size_t end
){
    std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>> splitdiag_vec{};
    splitdiag_vec.reserve(end-start);
    std::vector<DiagonalCut> diagcut_vec{};
    diagcut_vec.reserve(end-start);
    for(size_t i = start; i < end; i++){
        const auto [target_start, query_start, target_end, query_end] = *(diag_data+i);
        diagcut_vec.emplace_back(
            target_start,
            query_start,
            target_end,
            query_end
        );
    }
    for(size_t i = 0; i < diagcut_vec.size(); i++){
        auto& A = diagcut_vec[i];
        const auto [Ast, Asq] = A.start;
        const auto [Aet, Aeq] = A.end;
        for(size_t j = i+1; j < diagcut_vec.size(); j++){
            auto& B = diagcut_vec[j];
            const auto [Bst, Bsq] = B.start;
            const auto [Bet, Beq] = B.end;
            const bool overlapt = std::max(Ast, Bst) < std::min(Aet, Bet);
            const bool overlapq = std::max(Asq, Bsq) < std::min(Aeq, Beq);

            if(!overlapt || !overlapq) continue;

            const auto cut1 = std::max(Ast, Bst)-1;
            const auto cut2 = std::min(Aet, Bet);
            //std::cout << Ast << "\t" << Aet << "\t" << Bst << "\t" << Bet << "\t" << std::endl;
            //std::cout << cut1 << "\t" << cut2 << std::endl;
            if(Ast < cut1 && cut1 < Aet){
                A.insert(cut1);
                //std::cout << "Added Ast cut1 Aet" << std::endl;
            }
            if(Ast < cut2 && cut2 < Aet){
                A.insert(cut2);
                //std::cout << "Added Ast cut2 Aet" << std::endl;
            }
            if(Bst < cut1 && cut1 < Bet){
                B.insert(cut1);
                //std::cout << "Added Bst cut1 Bet" << std::endl;
            }
            if(Bst < cut2 && cut2 < Bet){
                B.insert(cut2);
                //std::cout << "Added Bst cut2 Bet" << std::endl;
            }
        }
    }
    for(const auto& cut: diagcut_vec){
        cut.merge_cut_diag(splitdiag_vec);
    }
    return splitdiag_vec;
}

std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>> score_diag(
    const std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t>>& diag_vec,
    const GttlMultiseq* target,
    const size_t target_seqnum,
    const GttlMultiseq* query,
    const size_t query_seqnum
) {
    std::vector<std::tuple<uint16_t,uint16_t,uint16_t,uint16_t,int32_t>> score_vec;
    const auto target_seq = target->sequence_ptr_get(target_seqnum);
    const auto query_seq = query->sequence_ptr_get(query_seqnum);
    for(const auto& diag : diag_vec){
        const auto target_start = std::get<0>(diag);
        const auto query_start = std::get<1>(diag);
        const auto target_end = std::get<2>(diag);
        const auto query_end = std::get<3>(diag);
        const auto matchlen = query_end - query_start + 1;
        const auto target_ptr = target_seq + target_start;
        const auto query_ptr = query_seq + query_start;
        int32_t score = 0;
        for(uint16_t i = 0; i < matchlen; i++){
            score += score_matrix[*(target_ptr + i)][*(query_ptr+i)];
        }
        score_vec.emplace_back(
            target_start,
            query_start,
            target_end,
            query_end,
            score
        );
    }
    return score_vec;
}

int main(int argc, char** argv) {
    SearchOptions options;

    try
    {
        options.parse(argc, argv);
    }
    catch (std::invalid_argument &e) /* check_err.py */
    {
        std::cerr << argv[0] << ": " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (options.help_option_is_set())
    {
        return EXIT_SUCCESS;
    }

    int faiss_ngpus = faiss::gpu::getNumDevices();
    std::cout << "Number of GPUs available for Faiss: " << faiss_ngpus << std::endl;

    int torch_ngpus = torch::cuda::device_count();
    std::cout << "Number of GPUs available for LibTorch: " << torch_ngpus << std::endl;

    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA runtime version: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    std::cout << "CUDA runtime version: " << major << "." << minor << std::endl;

    std::cout << "CUDA compile-time version: "
    << CUDA_VERSION / 1000 << "."
    << (CUDA_VERSION % 1000) / 10 << std::endl;

    constexpr const size_t min_unit_size=8;
    constexpr const size_t max_unit_size=9;
    const bool faiss_gpu = options.gpu_option_is_set();
    const bool ivf = options.ivf_option_is_set();
    const bool pq = options.pq_option_is_set();
    const size_t target_batch_size = std::stoi(options.target_batch_size_get());
    const size_t query_batch_size = std::stoi(options.query_batch_size_get());
    const auto model = get_model(options.model_path_get());
    const auto ndim = std::stoi(options.ndim_get());
    const auto nlist = std::stoi(options.nlist_get());
    const auto nprobe = std::stoi(options.nprobe_get());
    const auto subQuantizers = std::stoi(options.subquantizers_get());
    const auto bitsPerCode = std::stoi(options.bitsPerCode_get());
    const auto train_nsamples = std::stoi(options.train_nsamples_get());
    const auto k = std::stoi(options.knn_get());
    const auto threshold = std::stod(options.threshold_get());
    const auto tolerance = std::stoi(options.tolerance_get());
    const auto fraglen_bound = std::stoi(options.fraglen_bound_get());
    const auto gap_open = std::stod(options.gap_open_get());
    const auto gap_ext = std::stod(options.gap_ext_get());
    const auto mismatch = std::stod(options.mismatch_get());
    const bool show = options.show_option_is_set();
    const bool short_header = options.short_header_option_is_set();
    const std::string& index_path = options.import_index_get();
    const std::string& target_file = options.inputfiles_get()[0];
    const std::string& query_file = options.inputfiles_get()[1];
    auto target = new GttlMultiseq(target_file,true,UINT8_MAX);
    auto query = new GttlMultiseq(query_file,true,UINT8_MAX);
    const size_t target_seq_len_bits = target->sequences_length_bits_get();
    const size_t target_seq_num_bits = target->sequences_number_bits_get();
    const size_t query_seq_len_bits = query->sequences_length_bits_get();
    const size_t query_seq_num_bits = query->sequences_number_bits_get();
    const size_t diagonal_num_bits = gttl_required_bits<size_t>(target->sequences_maximum_length_get() + query->sequences_maximum_length_get());
    const auto sizeof_hit_unit = sizeof_unit_get(query_seq_num_bits+query_seq_len_bits+target_seq_num_bits+diagonal_num_bits);

    if (short_header){
        query->short_header_cache_create<'|','|'>();
        target->short_header_cache_create<'|','|'>();
    }
    auto literate_target = new LiterateMultiseq<char_spec,undefined_rank>(target);
    literate_target->perform_sequence_encoding();

    std::cout << "Model path: " << options.model_path_get() << std::endl;
    std::cout << "Index path: " << index_path << std::endl;
    std::cout << "GPU: " << (int) faiss_gpu << std::endl;
    std::cout << "IVF: " << (int) ivf << std::endl;
    std::cout << "PQ: " << (int) pq << std::endl;
    std::cout << "ndim: " << (int) ndim << std::endl;
    std::cout << "nlist: " << (int) nlist << std::endl;
    std::cout << "nprobe: " << (int) nprobe << std::endl;
    std::cout << "subQuantizers: " << (float) subQuantizers << std::endl;
    std::cout << "bitsPerCode: " << (int) bitsPerCode << std::endl;
    std::cout << "Number of training samples: " << (int) train_nsamples << std::endl;
    std::cout << "KNN: " << (int) k << std::endl;
    std::cout << "Threshold: " << (float) threshold << std::endl;
    std::cout << "Target batch size: " << (int) target_batch_size << std::endl;
    std::cout << "Query batch size: " << (int) query_batch_size << std::endl;
    std::cout << "Show: " << (int) show << std::endl;
    std::cout << "Short header: " << (int) short_header << std::endl;
    std::cout << "Tolerance: " << (int) tolerance << std::endl;
    std::cout << "Fragment length lower bound: " << (int) fraglen_bound << std::endl;
    std::cout << "Gap open: " << (float) gap_open << std::endl;
    std::cout << "Gap extend: " << (int) gap_ext << std::endl;
    std::cout << "Mismatch: " << (int) mismatch << std::endl;
    std::cout << "Size of query: " << (int) query->sequences_number_get() << std::endl;
    std::cout << "Size of target: " << (int) target->sequences_number_get() << std::endl;

    std::vector<Alignment> alignment_vec{};
    c10::InferenceMode guard;
    constexpr_for<0,2,1>([&] (auto constexpr_gpu){
        constexpr_for<0,2,1>([&] (auto constexpr_ivf){
            constexpr_for<0,2,1>([&] (auto constexpr_pq){
                if(constexpr_gpu == faiss_gpu && constexpr_ivf == ivf && constexpr_pq == pq){
                    RunTimeClass rt{};
                    /*faiss::Index* index;
                    faiss::Index* loaded_index = faiss::read_index(index_path.c_str());
                    std::unique_ptr<faiss::gpu::StandardGpuResources> res = std::make_unique<faiss::gpu::StandardGpuResources>();
                    if(constexpr_gpu){
                        index = faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, loaded_index);
                        delete loaded_index;
                    } else {
                        index = loaded_index;
                    }*/
                    using IndexClass = typename IndexClassSelector<constexpr_gpu, constexpr_ivf, constexpr_pq>::type;
                    IndexClass* index = new IndexClass(ndim);
                    index->import_index(index_path);
                    if constexpr(constexpr_ivf){
                        index->set_nprobe(nprobe);
                    }
                    const auto target_ids = reconstruct_id(
                        target
                    );
                    rt.show("Imported index");
                    std::cout << "Index of size " << index->index->ntotal << " with " <<  target_ids.size() << " vector ids" << std::endl;
                    const auto [triplets, query_ids] = kmer_query<constexpr_gpu, constexpr_ivf, constexpr_pq>(
                        model,
                        query,
                        query_batch_size,
                        index,
                        k,
                        threshold
                    );
                    rt.show("Query");
                    /*for(const auto& triplet : triplets){
			std::cout << std::get<0>(triplet) << "\t" << std::get<1>(triplet) << std::endl;
		    }*/
                    std::cout << "Query found " << triplets.size() << " significant matches with " <<  query_ids.size() << " vector ids" << std::endl;
                    //delete index;
                    constexpr_for<min_unit_size,max_unit_size+1,1>([&] (auto constexpr_hit_size){
                        if(constexpr_hit_size == sizeof_hit_unit){
                            const GttlBitPacker<constexpr_hit_size,4> match_packer{{
                                {static_cast<int>(target_seq_num_bits),
                                static_cast<int>(query_seq_num_bits),
                                static_cast<int>(diagonal_num_bits),
                                static_cast<int>(query_seq_len_bits)}}};
                            auto matches = triplet_to_hit<constexpr_hit_size>(
                                triplets,
                                query_ids,
                                target_ids,
                                match_packer,
                                query->sequences_maximum_length_get()
                            );
                            rt.show("Convert to hit");
                            hit_sort<constexpr_hit_size>(matches,target_seq_num_bits+query_seq_num_bits+diagonal_num_bits+query_seq_len_bits);
                            if(matches.size() > 0){
                                const size_t diag_bits = target_seq_num_bits+query_seq_num_bits+diagonal_num_bits+2*query_seq_len_bits;
                                const size_t sizeof_diag_unit = sizeof_unit_get(diag_bits);
                                constexpr_for<min_unit_size,max_unit_size+1,1>([&] (auto constexpr_diag_size){
                                    if(constexpr_diag_size == sizeof_diag_unit){
                                        const GttlBitPacker<constexpr_diag_size,5> diag_packer{{
                                            {static_cast<int>(target_seq_num_bits),
                                            static_cast<int>(query_seq_num_bits),
                                            static_cast<int>(diagonal_num_bits),
                                            static_cast<int>(query_seq_len_bits),
                                            static_cast<int>(query_seq_len_bits)}}};
                                        const auto diag_vec = match_to_diag<constexpr_hit_size, constexpr_diag_size>(
                                            matches,
                                            match_packer,
                                            diag_packer,
                                            tolerance,
                                            fraglen_bound
                                        );
                                        rt.show("Created diagonals");
                                        /*for(const auto& diag : diag_vec){
                                            * std::cout << diag.template decode_at<0>(diag_packer) << "\t" << diag.template decode_at<1>(diag_packer) << std::endl;
                                        }*/

                                        const auto [indexgroup, diaggroup] = create_groups<constexpr_diag_size>(
                                            diag_vec,
                                            diag_packer,
                                            query->sequences_maximum_length_get()
                                        );
                                        rt.show("Built groups");
                                        std::cout << "Found " << diaggroup.size() << " diagonals from " << indexgroup.size() << " sequence pair" << std::endl;

                                        for(const auto& [target_seqnum, query_seqnum, start_idx, end_idx] : indexgroup){
                                            /*std::cout << start_idx << "\t" << end_idx << std::endl;
                                            *                                            for(size_t i = start_idx; i < end_idx; i++){
                                            *                                                const auto& [st, sq, te, tq] = diaggroup[i];
                                            *                                                std::cout << st << "\t" << sq << "\t" << te << "\t" << tq << std::endl;
                                        }*/
                                            auto splitdiag = split_diag(
                                                diaggroup.data(),
                                                start_idx,
                                                end_idx
                                            );
                                            //std::cout << start_idx << "\t" << end_idx << std::endl;
                                            auto scored_diag = score_diag(
                                                splitdiag,
                                                target,
                                                target_seqnum,
                                                query,
                                                query_seqnum
                                            );
                                            /*for(const auto& [st, sq, te, tq, s] : scored_diag){
                                            *                                                std::cout << st << "\t" << sq << "\t" << te << "\t" << tq << "\t" << s << std::endl;
                                        }*/

                                            std::sort(scored_diag.begin(), scored_diag.end(), diag_comp);

                                            alignment_vec.emplace_back(
                                                target_seqnum,
                                                query_seqnum,
                                                scored_diag,
                                                gap_open,
                                                gap_ext,
                                                mismatch
                                            );
                                            //std::cout << "Finish alignment" << std::endl;
                                        }
                                        rt.show("Assembled alignment");

                                        std::sort(alignment_vec.begin(), alignment_vec.end(), score_comp);

                                        for(const auto& aln : alignment_vec){
                                            assert(aln.test());
                                        }

                                        if(show){
                                            std::cout << "Number of alignment: " << alignment_vec.size() << std::endl;
                                            if(!short_header){
                                                std::cout << "#target_seq_num" << '\t' << "query_seq_num" << '\t' << "target_len" << "\t" << "query_len" << "\t" << "aln_len" << "\t" <<
                                                "score" << "\t" << "num_diag" << '\t' << "target alignment" << '\t' << "query alignment"
                                                << std::endl;
                                                for(const auto& aln : alignment_vec){
                                                    const auto target_len = target->sequence_length_get(aln.target_seq);
                                                    const auto query_len = query->sequence_length_get(aln.query_seq);
                                                    std::cout << (int) aln.target_seq << '\t'
                                                    << (int) aln.query_seq << '\t' << target_len << "\t" << query_len << "\t" << aln.target_bits.count() << "\t" <<
                                                    aln.score << '\t' << aln.num_diag << "\t" <<
                                                    aln.target_bits.to_string().substr(0, target_len)
                                                    << '\t'
                                                    << aln.query_bits.to_string().substr(0, query_len)
                                                    <<  std::endl;
                                                }
                                            } else {
                                                std::cout << "#target_header" << '\t' << "query_header" << '\t' << "target_len" << "\t" << "query_len" << "\t" << "aln_len" << "\t" <<
                                                "score" << "\t" << "num_diag" << '\t' << "target alignment" << '\t' << "query alignment"
                                                << '\t' << std::endl;
                                                for(const auto& aln : alignment_vec){
                                                    const auto target_seq_num = aln.target_seq;
                                                    assert(target_seq_num < target->sequences_number_get());
                                                    const auto target_len = target->sequence_length_get(target_seq_num);
                                                    size_t target_sh_offset, target_sh_len;
                                                    std::tie(target_sh_offset,target_sh_len) = target->short_header_get(target_seq_num);
                                                    const std::string_view target_seq_header = target->header_get(target_seq_num);

                                                    const auto query_seq_num = aln.query_seq;
                                                    assert(query_seq_num < query->sequences_number_get());
                                                    const auto query_len = query->sequence_length_get(query_seq_num);
                                                    size_t query_sh_offset, query_sh_len;
                                                    std::tie(query_sh_offset,query_sh_len) = query->short_header_get(query_seq_num);
                                                    const std::string_view query_seq_header = query->header_get(query_seq_num);

                                                    std::cout << target_seq_header.substr(target_sh_offset,target_sh_len) << '\t'
                                                    << query_seq_header.substr(query_sh_offset,query_sh_len) << '\t' << target_len << "\t" << query_len  << "\t" << aln.target_bits.count() << "\t" <<
                                                    aln.score << "\t" << aln.num_diag << '\t' << aln.target_bits.to_string().substr(0, target->sequence_length_get(target_seq_num)) << '\t'
                                                    << aln.query_bits.to_string().substr(0, query->sequence_length_get(query_seq_num)) <<  std::endl;
                                                }
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    });
                }
            });
        });
    });
    delete literate_target;
    delete target;
    delete query;
    return 0;
}
