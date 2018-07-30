#ifndef CUSTOMSSD_H
#define CUSTOMSSD_H

#ifdef CUSTOMSSDDETECTION_EXPORTS
#define CUSTOMSSDDETECTION_API __declspec(dllexport)
#else
#define CUSTOMSSDDETECTION_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

	typedef struct CustomSSD_ctx CustomSSD_ctx;

	CUSTOMSSDDETECTION_API void GlobalInitCustomSSD(char*** pargv);

	CUSTOMSSDDETECTION_API CustomSSD_ctx* CustomSSD_initialize(const char* model_file, const char* trained_file,
		const char* mean_file, const char* mean_value,
		const char* label_file, float iou_threshold,
		float ios_threshold, float conf_threshold);

	CUSTOMSSDDETECTION_API const char* CustomSSD_inference(CustomSSD_ctx* ctx,
		char* buffer, size_t length);

	CUSTOMSSDDETECTION_API void CustomSSD_destroy(CustomSSD_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
