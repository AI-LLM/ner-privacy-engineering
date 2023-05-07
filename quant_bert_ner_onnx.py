from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("bert_ner_ft.onnx", "bert_ner_int8.onnx", 
                 weight_type=QuantType.QUInt8)
