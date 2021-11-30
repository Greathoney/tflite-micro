#include <math.h>

// #include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
// #include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h" 

#include "tensorflow/lite/micro/examples/keyboard/model_settings.h"
#include "tensorflow/lite/micro/examples/keyboard/hand_data.h"
#include "tensorflow/lite/micro/examples/keyboard/hand_landmark_full.h"


const int tensorArenaSize = 100*1024;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
        // Set up logging.

    tflite::MicroErrorReporter micro_error_reporter;

    const tflite::Model* model = ::tflite::GetModel(hand_landmark_full_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(&micro_error_reporter,
                            "Model provided is schema version %d not equal "
                            "to supported version %d.\n",
                            model->version(), TFLITE_SCHEMA_VERSION);
    }

    tflite::AllOpsResolver resolver;

    constexpr int kTensorArenaSize = tensorArenaSize;
    uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                        kTensorArenaSize, &micro_error_reporter);
    TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);


    TfLiteTensor* input = interpreter.input(0);

    TF_LITE_MICRO_EXPECT_NE(nullptr, input);
    TF_LITE_MICRO_EXPECT_EQ(3, input->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
    TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[1]);
    // TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);


    memcpy(input->data.int8, g_hand_data, input->bytes);

    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    TfLiteTensor* output = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
    // TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

    float posZ = output->data.f[11];



    TF_LITE_REPORT_ERROR(&micro_error_reporter, "finger: %f\n", (double)posZ);


    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Ran successfully\n");

}

TF_LITE_MICRO_TESTS_END
