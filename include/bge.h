#include "ax_devices.h"
#ifndef __BGE_H__
#define __BGE_H__

#ifdef __cplusplus
extern "C"
{
#endif
#define MAX_TOKENS 512
#define TOKEN_FEATURE_DIM 384

    typedef struct
    {
        ax_devive_e dev_type;
        int devid;
        
        char filename_axmodel[1024];
    } embeding_attr_t;

    typedef struct
    {
        int len_of_tokens;
        float embeding[TOKEN_FEATURE_DIM];
    } embeding_t;

    typedef void *embeding_handle_t;

    int ax_embeding_init(embeding_attr_t *attr, embeding_handle_t *handle);
    int ax_embeding_deinit(embeding_handle_t handle);

    int ax_embeding(embeding_handle_t handle, char *text, embeding_t *embeding);

    float ax_similarity(embeding_t *embeding_1, embeding_t *embeding_2);

#ifdef __cplusplus
}
#endif

#endif
