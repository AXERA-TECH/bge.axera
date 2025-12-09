#include "bge.h"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "sample_log.h"

#include <memory>
#include "mmap.hpp"
#include "tokenizer/tokenizer.hpp"
#include <cmath>

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

#define CLS_TOKEN 101
#define SEP_TOKEN 102
#define PAD_TOKEN 0

struct embeding_handle_internal_t
{
    std::shared_ptr<ax_runner_base> runner;
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer;
};

int ax_embeding_init(embeding_attr_t *init_info, embeding_handle_t *handle)
{
    if (init_info->dev_type == ax_devive_e::host_device)
    {
        if (!get_ax_sys_loader().is_init() || !get_ax_engine_loader().is_init())
        {
            ALOGE("axsys or axengine not init");
            return -1;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        if (!getLoader().is_init())
        {
            ALOGE("unsupport axcl");
            return -1;
        }

        if (!axcl_Dev_IsInit(init_info->devid))
        {
            ALOGE("axcl device %d not init", init_info->devid);
            return -1;
        }
    }
    else
    {
        return -1;
    }

    embeding_handle_internal_t *internal = new embeding_handle_internal_t();

    MMap model_mmap(init_info->filename_axmodel);

    std::shared_ptr<ax_runner_base> runner;
    if (init_info->dev_type == ax_devive_e::host_device)
    {
        runner = std::make_shared<ax_runner_ax650>();
        auto ret = runner->init(model_mmap.data(), model_mmap.size(), -1);
        if (ret != 0)
        {
            ALOGE(" init failed");
            return -1;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        runner = std::make_shared<ax_runner_axcl>();
        auto ret = runner->init(model_mmap.data(), model_mmap.size(), init_info->devid);
        if (ret != 0)
        {
            ALOGE(" init failed");
            return -1;
        }
    }
    else
    {
        printf("unsupport dev type\n");
        return -1;
    }

    internal->runner = runner;

    internal->tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(init_info->tokenizer_model));
    if (internal->tokenizer == nullptr)
    {
        ALOGE("create tokenizer failed");
        return -1;
    }

    *handle = internal;

    return 0;
}

int ax_embeding_deinit(embeding_handle_t handle)
{
    if (handle == nullptr)
    {
        return 0;
    }
    embeding_handle_internal_t *internal = (embeding_handle_internal_t *)handle;
    internal->runner->deinit();
    delete internal;
    return 0;
}

int ax_embeding(embeding_handle_t handle, char *text, embeding_t *embeding)
{
    if (embeding == nullptr)
    {
        return -1;
    }
    embeding_handle_internal_t *internal = (embeding_handle_internal_t *)handle;

    std::vector<int> _token_ids;
    _token_ids = internal->tokenizer->encode(text);
    if (_token_ids.size() > MAX_TOKENS)
    {
        ALOGW("text len %d > MAX_TOKENS %d, truncate to %d", _token_ids.size(), MAX_TOKENS, MAX_TOKENS);
        _token_ids.resize(MAX_TOKENS);
    }

    _token_ids.insert(_token_ids.begin(), CLS_TOKEN);
    _token_ids.push_back(SEP_TOKEN);

    memset(internal->runner->get_input(0).pVirAddr, 0, internal->runner->get_input(0).nSize);
    memcpy(internal->runner->get_input(0).pVirAddr, _token_ids.data(), _token_ids.size() * sizeof(int));

    internal->runner->inference();

    embeding->len_of_tokens = _token_ids.size();
    memcpy(embeding->embeding, internal->runner->get_output(0).pVirAddr, TOKEN_FEATURE_DIM * sizeof(float));

    float norm = 0.0f;
    for (int i = 0; i < TOKEN_FEATURE_DIM; i++)
        norm += embeding->embeding[i] * embeding->embeding[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < TOKEN_FEATURE_DIM; i++)
        embeding->embeding[i] /= norm;
    return 0;
}

float ax_similarity(embeding_t *embeding_1, embeding_t *embeding_2)
{
    if (embeding_1 == nullptr || embeding_2 == nullptr)
    {
        return -1;
    }
    float sim = 0.0f;
    for (int i = 0; i < TOKEN_FEATURE_DIM; i++)
        sim += embeding_1->embeding[i] * embeding_2->embeding[i];
    sim = sim < 0 ? 0 : sim > 1 ? 1 : sim;
    return sim;
}
