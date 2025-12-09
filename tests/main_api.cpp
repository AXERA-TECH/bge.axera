#include "bge.h"
#include "utils/cmdline.hpp"
#include "utils/httplib.h"
#include "utils/json.hpp"
#include <fstream>
#include <cstring>
#include <signal.h>

httplib::Server svr;

void _signal(int signo)
{
    svr.stop();
}

int main(int argc, char *argv[])
{
    signal(SIGINT, _signal);
    signal(SIGTERM, _signal);

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
    parser.add<int>("port", 'p', "port number", false, 8080);
    parser.parse_check(argc, argv);

    sprintf(init_info.filename_axmodel, "%s", parser.get<std::string>("model").c_str());

    printf("filename_axmodel: %s\n", init_info.filename_axmodel);

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

    svr.Post("/embedding", [&](const httplib::Request &req, httplib::Response &res)
             {
                 try
                 {
                     std::string sentence = nlohmann::json::parse(req.body)["sentence"];
                     embeding_t embeding;
                     ax_embeding(handle, (char *)sentence.c_str(), &embeding);

                     nlohmann::json json;
                     json["embedding"] = embeding.embeding;

                     res.set_content(json.dump(), "application/json");
                 }
                 catch (const std::exception &e)
                 {
                     res.set_content(e.what(), "text/plain");
                     return;
                 } });

    printf("listen http://0.0.0.0:%d\n", parser.get<int>("port"));
    printf("for example: \n    curl -X POST http://0.0.0.0:%d/embedding -H \"Content-Type: application/json\" -d '{\"sentence\":\"I really love math\"}'\n", parser.get<int>("port"));
    printf("press Ctrl+C to stop\n");
    svr.listen("0.0.0.0", parser.get<int>("port"));

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