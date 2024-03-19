# import ctcdecode

from module.text_process import TextProcess

textprocess = TextProcess()

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]


def convert_to_string(tokens, vocab, seq_len):
    return ''.join([vocab[x] for x in tokens[0:seq_len]])


class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        self.blank_id = blank_id

        print("loading beam search with lm...")
        # self.decoder = ctcdecode.CTCBeamDecoder(
        #     labels, alpha=0.522729216841, beta=0.96506699808,
        #     beam_width=beam_size, blank_id=labels.index('_'),
        #     model_path=kenlm_path)
        print("finished loading beam search")

    def __call__(self, output):
        # beam_result, beam_scores, timestamps, out_seq_len = self.decoder.decode(output)
        # return convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])
        return output
