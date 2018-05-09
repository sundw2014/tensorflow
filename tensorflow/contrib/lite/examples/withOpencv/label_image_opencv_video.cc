// https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/examples/label_image/label_image.cc
//
#include <iostream>
#include "opencv2/opencv.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "tensorflow/contrib/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/contrib/lite/examples/label_image/get_top_n.h"

#define LOG(x) std::cerr

namespace tflite {
namespace label_image {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void display_usage() {
  LOG(INFO) << "label_image\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--image, -i: image_name.bmp\n"
            << "--labels, -l: labels for the model\n"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--profiling, -p: [0|1], profiling or not\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  // parsing options
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, 0, 'a'},
        {"count", required_argument, 0, 'c'},
        {"verbose", required_argument, 0, 'v'},
        {"image", required_argument, 0, 'i'},
        {"labels", required_argument, 0, 'l'},
        {"tflite_model", required_argument, 0, 'm'},
        {"profiling", required_argument, 0, 'p'},
        {"threads", required_argument, 0, 't'},
        {"input_mean", required_argument, 0, 'b'},
        {"input_std", required_argument, 0, 's'},
        {0, 0, 0, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:i:l:m:p:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, (char**)NULL, 10);
        break;
      case 'b':
        s.input_mean = strtod(optarg, NULL);
        break;
      case 'c':
        s.loop_count = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, (char**)NULL, 10);
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, (char**)NULL, 10);
        break;
      case 's':
        s.input_std = strtod(optarg, NULL);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, (char**)NULL, 10);
        break;
      case 'v':
        s.verbose = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, (char**)NULL, 10);
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  // end of parsing options

  // setting up the opencv context
  // Create a VideoCapture object and use camera to capture the video
  cv::VideoCapture cap(0);

  // Check if camera opened successfully
  if(!cap.isOpened())
  {
    std::cout << "Error opening video stream" << std::endl;
    return -1;
  }

  // Default resolution of the frame is obtained.The default resolution is
  // system dependent.
  int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  LOG(INFO) << "camera resolution: (w x h) " << frame_width << " x " << frame_height << std::endl;

  // Define the codec and create VideoWriter object.The output is stored in
  // 'outcpp.avi' file.
  cv::VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10,
                        cv::Size(frame_width, frame_height));

  // end of setting up the opencv context

  // setting up the tf-lite context
  if (!s.model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s.model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s.model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << s.model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder (*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s.accel);

  if (s.verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s.number_of_threads != -1) {
    interpreter->SetNumThreads(s.number_of_threads);
  }

  int input = interpreter->inputs()[0];
  if (s.verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (s.verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (s.verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  // profiling::Profiler* profiler = new profiling::Profiler();
  // interpreter->SetProfiler(profiler);

  const int output_size = 1000;
  const size_t num_results = 5;
  const float threshold = 0.001f;

  int output = interpreter->outputs()[0];

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(s.labels_file_name, &labels, &label_count) != kTfLiteOk)
    exit(-1);
  // end of setting up the tf-lite context

  // main loop
  while (1) {
    // Capture frame-by-frame
    cv::Mat frame;
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty()) break;

    //TODO: center crop
    int image_width = frame_width;
    int image_height = frame_height;
    int image_channels = 3;

    // convert to array
    std::vector<uint8_t> array;
    if (frame.isContinuous()) {
      array.assign(frame.datastart, frame.dataend);
    } else {
      for (int i = 0; i < frame.rows; ++i) {
        array.insert(array.end(), frame.ptr<uint8_t>(i),
                     frame.ptr<uint8_t>(i) + frame.cols);
      }
    }

    uint8_t* in = array.data();

    // inference
    switch (interpreter->tensor(input)->type) {
      case kTfLiteFloat32:
        s.input_floating = true;
        resize<float>(interpreter->typed_tensor<float>(input), in, image_height,
                      image_width, image_channels, wanted_height, wanted_width,
                      wanted_channels, &s);
        break;
      case kTfLiteUInt8:
        resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in,
                        image_height, image_width, image_channels,
                        wanted_height, wanted_width, wanted_channels, &s);
        break;
      default:
        LOG(FATAL) << "cannot handle input type "
                   << interpreter->tensor(input)->type << " yet";
        exit(-1);
    }

    // if (s.profiling) profiler->StartProfiling();

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
    gettimeofday(&stop_time, NULL);
    // LOG(INFO) << "invoked \n";
    LOG(INFO) << "time: " << (get_us(stop_time) - get_us(start_time)) / (1000)
              << " ms \n";

    // if (s.profiling) {
    //   profiler->StopProfiling();
    //   auto profile_events = profiler->GetProfileEvents();
    //   for (int i = 0; i < profile_events.size(); i++) {
    //     auto op_index = profile_events[i]->event_metadata;
    //     const auto node_and_registration =
    //         interpreter->node_and_registration(op_index);
    //     const TfLiteRegistration registration =
    //     node_and_registration->second;
    //     PrintProfilingInfo(profile_events[i], op_index, registration);
    //   }
    // }

    std::vector<std::pair<float, int>> top_results;

    switch (interpreter->tensor(output)->type) {
      case kTfLiteFloat32:
        get_top_n<float>(interpreter->typed_output_tensor<float>(0),
                         output_size, num_results, threshold, &top_results,
                         true);
        break;
      case kTfLiteUInt8:
        get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                           output_size, num_results, threshold, &top_results,
                           false);
        break;
      default:
        LOG(FATAL) << "cannot handle output type "
                   << interpreter->tensor(input)->type << " yet";
        exit(-1);
    }

    // for (const auto& result : top_results) {
    //   const float confidence = result.first;
    //   const int index = result.second;
    //   LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
    // }

    float y0 = 50;
    float dy = 30;
    for (int i = 0; i < 5; i++) {
      std::ostringstream text;
      text << labels[top_results[i].second] << ": " << top_results[i].first;
      float y = y0 + i * dy;
      cv::putText(frame, text.str(), cv::Point(50, y), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0, 0, 255));
    }

    // Write the frame into the file 'outcpp.avi'
    video.write(frame);

    // Display the resulting frame
    cv::imshow("Frame", frame);

    // Press  ESC on keyboard to  exit
    char c = (char)cv::waitKey(1);
    if (c == 27) break;
  }
  // When everything done, release the video capture and write object
  cap.release();
  video.release();

  // Closes all the windows
  cv::destroyAllWindows();
  return 0;
}
}
}

int main(int argc, char* argv[]) {
  return tflite::label_image::Main(argc, argv);
}
