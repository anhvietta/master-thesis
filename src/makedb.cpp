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
    export_index = "",//Index setup
    train_file="",
    ndim = "320", nlist = "20000", subQuantizers = "40", bitsPerCode = "8", //Index options
    train_nsamples = "10000",
    target_batch_size = "1000" //Torch related
    ;

public:
    SearchOptions() {};

    void parse(int argc, char **argv)
    {
        cxxopts::Options options(argv[0],"Encode FASTA file and package into an index");
        options.set_width(80);
        options.custom_help(std::string("[options] target_db"));
        options.set_tab_expansion();
        options.add_options()
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
        ("subquantizers", "PQ Options: number of subquantizers",
         cxxopts::value<std::string>(subQuantizers)->default_value("40"))
        ("bitspercode", "PQ Options: bits per encoding dimension",
         cxxopts::value<std::string>(bitsPerCode)->default_value("8"))
        ("train_samples", "IVF/PQ: Number of training sequences",
         cxxopts::value<std::string>(train_nsamples)->default_value("10000"))
        ("train_file", "IVF/PQ: Train fasta file",
         cxxopts::value<std::string>(train_file)->default_value(""))
        ("export_index", "export index",
         cxxopts::value<std::string>(export_index)->default_value(""))
        ("b,target_batch_size", "set batch size when creating index",
         cxxopts::value<std::string>(target_batch_size)->default_value("1000"))
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
                if(unmatched_args.size() != 1){
                    throw cxxopts::OptionException("Exact 1 fasta file is needed");
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
    const std::string &target_batch_size_get(void) const noexcept
    {
        return target_batch_size;
    }
    const std::string &train_file_get(void) const noexcept
    {
        return train_file;
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

template<bool gpu>
class FlatIndex {
    using IndexType = std::conditional_t<gpu,
    faiss::gpu::GpuIndexFlatIP,
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
            //cfg->useFloat16 = true;
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

    /*void import_index(const std::string& import_path) {
     *       if constexpr(gpu){
     *           ExportType* e_index = faiss::read_index(import_path.c_str());
     *           res = std::make_unique<faiss::gpu::StandardGpuResources>();
     *           index = faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, e_index);
     *           delete e_index;
} else {
    index = faiss::read_index(import_path.c_str());
}
}*/

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

    void init(){
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            cfg = new ConfigType();
            //cfg->useFloat16 = true;
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

    void set_pq_params(const size_t subQuantizers_, const size_t bitsPerCode_){
        subQuantizers = subQuantizers_;
        bitsPerCode = bitsPerCode_;
    }

    void init(){
        if constexpr(gpu){
            res = std::make_unique<faiss::gpu::StandardGpuResources>();
            cfg = new ConfigType();
            //cfg->useFloat16 = true;
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

    /*void import_index(const std::string& import_path) {
     *       if constexpr(gpu){
     *           ExportType* e_index = faiss::read_index(import_path.c_str());
     *           res = std::make_unique<faiss::gpu::StandardGpuResources>();
     *           index = faiss::gpu::index_cpu_to_gpu(res.get(), device_idx, e_index);
     *           delete e_index;
} else {
    index = faiss::read_index(import_path.c_str());
}
}*/

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
    index->index->train(v_accumulator.size() / ndim, v_accumulator.data());
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
    const auto model = get_model(options.model_path_get());
    const auto ndim = std::stoi(options.ndim_get());
    const auto nlist = std::stoi(options.nlist_get());
    const auto subQuantizers = std::stoi(options.subquantizers_get());
    const auto bitsPerCode = std::stoi(options.bitsPerCode_get());
    const auto train_nsamples = std::stoi(options.train_nsamples_get());
    const auto train_file = options.train_file_get();
    const std::string& index_path = options.export_index_get();
    const std::string& target_file = options.inputfiles_get()[0];
    auto target = new GttlMultiseq(target_file,true,UINT8_MAX);

    std::cout << "Model path: " << options.model_path_get() << std::endl;
    std::cout << "GPU: " << (int) faiss_gpu << std::endl;
    std::cout << "IVF: " << (int) ivf << std::endl;
    std::cout << "PQ: " << (int) pq << std::endl;
    std::cout << "ndim: " << (int) ndim << std::endl;
    std::cout << "nlist: " << (int) nlist << std::endl;
    std::cout << "subQuantizers: " << (float) subQuantizers << std::endl;
    std::cout << "bitsPerCode: " << (int) bitsPerCode << std::endl;
    std::cout << "Number of training samples: " << (int) train_nsamples << std::endl;
    std::cout << "Target batch size: " << (int) target_batch_size << std::endl;
    std::cout << "Size of target: " << (int) target->sequences_number_get() << std::endl;

    constexpr_for<0,2,1>([&] (auto constexpr_gpu){
        constexpr_for<0,2,1>([&] (auto constexpr_ivf){
            constexpr_for<0,2,1>([&] (auto constexpr_pq){
                if(constexpr_gpu == faiss_gpu && constexpr_ivf == ivf && constexpr_pq == pq){
                    using IndexClass = typename IndexClassSelector<constexpr_gpu, constexpr_ivf, constexpr_pq>::type;
                    IndexClass* index = new IndexClass(ndim);
                    if constexpr(constexpr_pq){
                        index->set_pq_params(subQuantizers, bitsPerCode);
                    }
                    //std::cout << "set PQ params" << std::endl;
                    if constexpr(constexpr_ivf){
                        index->set_nlist(nlist);
                    }
                    std::cout << "init" << std::endl;
                    index->init();
                    RunTimeClass rt{};
                    if constexpr(constexpr_ivf){
                        GttlMultiseq* train = new GttlMultiseq(train_file,true,UINT8_MAX);
                        train_index<constexpr_gpu, constexpr_ivf, constexpr_pq>(
                            model,
                            train,
                            target_batch_size,
                            index,
                            train_nsamples,
                            ndim
                        );
                        delete train;
                    }
                    rt.show("Train index");
                    const auto target_ids = build_index<constexpr_gpu, constexpr_ivf, constexpr_pq>(
                        model,
                        target,
                        target_batch_size,
                        index
                    );
                    rt.show("Built index");
                    std::cout << "Index of size " << index->index->ntotal << " with " <<  target_ids.size() << " vector ids" << std::endl;
                    index->export_index(index_path);
                    std::cout << "Saved index at " << index_path << std::endl;
                    delete index;
                }
            });
        });
    });
    delete target;
    return 0;
}
