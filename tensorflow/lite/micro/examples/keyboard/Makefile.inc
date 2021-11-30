EXAMPLE_NAME:=keyboard

KEYBOARD_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/hand_landmark_full.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/hand_data.cc 


OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler_test.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc 

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h  

KEYBOARD_SRCS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.cc \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.cc 

KEYBOARD_HDRS := \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/output_handler.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/constants.h \
tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/main_functions.h 

KEYBOARD_GENERATOR_INPUTS := \
tensorflow/lite/micro/examples/keyboard/hand_landmark_full.tflite

KEYBOARD_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/keyboard/hand_landmark_full.cc

KEYBOARD_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)tensorflow/lite/micro/examples/keyboard/hand_landmark_full.h

#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/$(EXAMPLE_NAME)/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,$(EXAMPLE_NAME)_test,\
$(KEYBOARD_TEST_SRCS),,$(KEYBOARD_GENERATOR_INPUTS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))

# Builds a standalone binary.
$(eval $(call microlite_test,$(EXAMPLE_NAME),\
$(KEYBOARD_SRCS),$(KEYBOARD_HDRS),$(KEYBOARD_GENERATOR_INPUTS)))

# Add sources and headers generated from $(KEYBOARD_GENERATOR_INPUTS).
KEYBOARD_SRCS += $(KEYBOARD_GENERATED_SRCS)
KEYBOARD_HDRS += $(KEYBOARD_GENERATED_HDRS)

list_$(EXAMPLE_NAME)_example_sources:
	@echo $(KEYBOARD_SRCS)

list_$(EXAMPLE_NAME)_example_headers:
	@echo $(KEYBOARD_HDRS)
