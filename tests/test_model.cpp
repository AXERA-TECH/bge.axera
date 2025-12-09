#include "bge.h"
#include "utils/cmdline.hpp"
#include <fstream>
#include <cstring>

int main(int argc, char *argv[])
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }
    else
    {
        printf("no device available\n");
        return -1;
    }
    embeding_attr_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "encoder model(onnx model or axmodel)", true);
    parser.add<std::string>("tokenizer", 't', "tokenizer path", true);
    parser.parse_check(argc, argv);

    sprintf(init_info.filename_axmodel, "%s", parser.get<std::string>("model").c_str());
    sprintf(init_info.tokenizer_model, "%s", parser.get<std::string>("tokenizer").c_str());

    printf("filename_axmodel: %s\n", init_info.filename_axmodel);
    printf("tokenizer_model: %s\n", init_info.tokenizer_model);

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    embeding_handle_t handle;
    int ret = ax_embeding_init(&init_info, &handle);
    if (ret != 0)
    {
        printf("ax_embeding_init failed\n");
        return -1;
    }

    std::vector<std::string> sentences_1 = {"I really love math", "so do I"};
    std::vector<std::string> sentences_2 = {"I pretty like mathematics", "same as me"};

    for (int i = 0; i < sentences_1.size(); i++)
    {
        embeding_t embeding_1, embeding_2;
        ax_embeding(handle, (char *)sentences_1[i].c_str(), &embeding_1);
        for (int j = 0; j < sentences_2.size(); j++)
        {
            ax_embeding(handle, (char *)sentences_2[j].c_str(), &embeding_2);
            float sim = ax_similarity(&embeding_1, &embeding_2);
            printf("similarity between \33[32m%s\33[0m and \33[34m%s\33[0m is %f\n", sentences_1[i].c_str(), sentences_2[j].c_str(), sim);
        }
    }

    ax_embeding_deinit(handle);

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}