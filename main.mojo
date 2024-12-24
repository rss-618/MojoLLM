#!/usr/bin/env mojo
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from max.engine import InputSpec, InferenceSession, Model
from pathlib import Path
from python import Python, PythonObject
from max.tensor import Tensor, TensorSpec
import sys


fn execute(
    model: Model, text: String, transformers: PythonObject
) raises -> String:
    # The model was compiled with a maximum seqlen, so read that out from the model output metadata
    var output_spec = model.get_model_output_metadata()[0]
    var max_seqlen = output_spec[1].value()

    var tokenizer = transformers.AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    var inputs = tokenizer(
        text=text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_seqlen,
        return_tensors="np",
    )

    var input_ids = inputs["input_ids"]
    var token_type_ids = inputs["token_type_ids"]
    var attention_mask = inputs["attention_mask"]

    var outputs = model.execute(
        "input_ids",
        input_ids,
        "token_type_ids",
        token_type_ids,
        "attention_mask",
        attention_mask,
    )

    var logits = outputs.get[DType.float32]("result0")

    var mask_idx = -1
    for i in range(len(input_ids[0])):
        if input_ids[0][i] == tokenizer.mask_token_id:
            mask_idx = i

    var predicted_token_id = argmax(logits)[mask_idx]
    return str(
        tokenizer.decode(
            predicted_token_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )


def argmax(t: Tensor) -> List[Int]:
    var res = List[Int](capacity=t.dim(1))
    for i in range(t.dim(1)):
        var max_val = Scalar[t.type].MIN
        var max_idx = 0
        for j in range(t.dim(2)):
            if t[0, i, j] > max_val:
                max_val = t[0, i, j]
                max_idx = j
        res.append(max_idx)
    return res


def load_model(session: InferenceSession) -> Model:
    var batch = 1
    var seqlen = 128

    var input_ids_spec = TensorSpec(DType.int64, batch, seqlen)
    var token_type_ids_spec = TensorSpec(DType.int64, batch, seqlen)
    var attention_mask_spec = TensorSpec(DType.int64, batch, seqlen)
    var input_specs = List[InputSpec]()

    input_specs.append(input_ids_spec)
    input_specs.append(attention_mask_spec)
    input_specs.append(token_type_ids_spec)

    var model = session.load(
        Path("bert-mlm.torchscript"), input_specs=input_specs
    )

    return model


fn read_input() raises -> String:
    var USAGE = 'Usage: ./main <str> \n\t e.g., ./main "Paris is the [MASK] of France"'

    var argv = sys.argv()
    if len(argv) != 2:
        raise Error("\nPlease enter a prompt." + "\n" + USAGE)

    return sys.argv()[1]


fn main():
    try:
        # Import HF Transformers dependency (for the tokenizer)
        var transformers = Python.import_module("transformers")

        # Read user prompt, create an InferenceSession, and load the model
        var text = read_input()
        var session = InferenceSession()
        var model = load_model(session)

        # Run inference
        var decoded_result = execute(model, text, transformers)

        print("input text: ", text)
        print("filled mask: ", text.replace("[MASK]", decoded_result))
    except:
        print("An error has ocurred.")
