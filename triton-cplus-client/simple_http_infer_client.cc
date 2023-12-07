#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "http_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t-i <none|gzip|deflate>" << std::endl;
  std::cerr << "\t-o <none|gzip|deflate>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "\t--verify-peer" << std::endl;
  std::cerr << "\t--verify-host" << std::endl;
  std::cerr << "\t--ca-certs" << std::endl;
  std::cerr << "\t--cert-file" << std::endl;
  std::cerr << "\t--key-file" << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl
      << "For -i, it sets the compression algorithm used for sending request "
         "body."
      << "For -o, it sets the compression algorithm used for receiving "
         "response body."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  auto request_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  auto response_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  long verify_peer = 1;
  long verify_host = 2;
  std::string cacerts;
  std::string certfile;
  std::string keyfile;

  static struct option long_options[] = {
      {"verify-peer", 1, 0, 0}, {"verify-host", 1, 0, 1}, {"ca-certs", 1, 0, 2},
      {"cert-file", 1, 0, 3},   {"key-file", 1, 0, 4},    {0, 0, 0, 0}};

  int opt;
  while((opt = getopt_long(argc, argv, "v:u:t:H:i:o:", long_options, NULL)) !=
        -1) {
    switch (opt) {
      case 0:
        verify_peer = std::atoi(optarg);
        break;
      case 1:
        verify_host = std::atoi(optarg);
        break;
      case 2:
        cacerts = optarg;
        break;
      case 3:
        certfile = optarg;
        break;
      case 4:
        keyfile = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'i': {
        std::string arg = optarg;
        if (arg == "gzip") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case 'o': {
        std::string arg = optarg;
        if (arg == "gzip") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  std::string model_name = "simple";
  std::string model_version = "";

  tc::HttpSslOptions ssl_options;
  ssl_options.verify_peer = verify_peer;
  ssl_options.verify_host = verify_host;
  ssl_options.ca_info = cacerts;
  ssl_options.cert = certfile;
  ssl_options.key = keyfile;

  std::unique_ptr<tc::InferenceServerHttpClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerHttpClient::Create(&client, url, verbose, ssl_options),
      "unable to create http client");

  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  std::vector<int64_t> shape{1, 16};

  tc::InferInput* input0;
  tc::InferInput* input1;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<tc::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input1_data[0]),
          input1_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {};

  tc::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(
          &results, options, inputs, outputs, http_headers, tc::Parameters(),
          request_compression_algorithm, response_compression_algorithm),
      "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  ValidateShapeAndDatatype("OUTPUT0", results_ptr);
  ValidateShapeAndDatatype("OUTPUT1", results_ptr);

  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  int32_t* output1_data;
  size_t output1_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
      "unable to get result data for 'OUTPUT1'");
  if (output1_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
              << output1_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input0_data[i] << " = "
              << *(output0_data + i) << std::endl;
    std::cout << input0_data[i] << " - " << input0_data[i] << " = "
              << *(output1_data + i) << std::endl;
    if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  std::cout << results_ptr->DebugString() << std::endl;

  tc::InferStat infer_stat;
  client->ClientInferStat(&infer_stat);
  std::cout << "======Client Statistics======" << std::endl;
  std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
  std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
  std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
  std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

  std::string model_stat;
  FAIL_IF_ERR(
      client->ModelInferenceStatistics(&model_stat, model_name),
      "unable to get model statistics");
  std::cout << "======Model Statistics======" << std::endl;
  std::cout << model_stat << std::endl;

  std::cout << "PASS : Infer" << std::endl;

  return 0;
}
