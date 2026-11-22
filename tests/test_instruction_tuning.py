"""Offline checks with a tiny random Llama and a synthetic byte-level tokenizer.

These test mechanics, not the downloaded SmolLM2 tokenizer or book quality.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM, DataCollatorForSeq2Seq

from tokipona_torch.data import encode, inspect_example, chat_messages, load_dataset
from tokipona_torch.model import add_lora, load_adapter
from tokipona_torch.pilot import PilotConfig, run_pilot
from scripts.prepare_book import clean

# Published SmolLM2 chat template structure; vocabulary below is synthetic.
# https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/tokenizer_config.json
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def fixture_tokenizer():
    raw=Tokenizer(models.BPE(unk_token='<unk>'))
    raw.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder=decoders.ByteLevel()
    raw.train_from_iterator(['Hello world. Question? A useful answer. You are a helpful AI assistant.']*3,
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=['<unk>','<|im_start|>','<|im_end|>']))
    return PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='<unk>',bos_token='<|im_start|>',
        eos_token='<|im_end|>',pad_token='<|im_end|>',chat_template=TEMPLATE,model_max_length=1024)


class InstructionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer=fixture_tokenizer()

    def row(self):
        return {'messages':[{'role':'system','content':'Keep it brief.'},
            {'role':'user','content':'First question?'},{'role':'assistant','content':'Earlier answer.'},
            {'role':'user','content':'Question?'},{'role':'assistant','content':'A useful answer.'}]}

    def model(self):
        return LlamaForCausalLM(LlamaConfig(vocab_size=len(self.tokenizer),hidden_size=32,
            intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=1024,bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))

    def test_system_and_final_response_mask(self):
        self.assertEqual(chat_messages(self.row()['messages'],self.tokenizer)[0]['role'],'system')
        view=inspect_example(self.row(),self.tokenizer,512)
        self.assertIn('A useful answer.',view['target_text'])
        self.assertNotIn('Earlier answer.',view['target_text'])
        self.assertNotIn('Question?',view['target_text'])
        self.assertTrue(view['eos_supervised'])
        self.assertFalse(view['truncated'])
        boundary=view['response_character_boundary']
        self.assertIn('assistant',view['rendered_chat'][:boundary])

    def test_padding_does_not_mask_real_eos(self):
        first=encode(self.row(),self.tokenizer,512,'response')
        second=encode({'messages':[{'role':'user','content':'Hello'},
            {'role':'assistant','content':'world'}]},self.tokenizer,512,'response')
        batch=DataCollatorForSeq2Seq(self.tokenizer)([first,second])
        for i, row in enumerate([first,second]):
            n=len(row['input_ids'])
            self.assertTrue(torch.all(batch['labels'][i,n:]==-100))
            self.assertTrue(any(token==self.tokenizer.eos_token_id and label!=-100
                for token,label in zip(batch['input_ids'][i,:n],batch['labels'][i,:n])))

    def test_truncation_and_draft_rejection(self):
        self.assertIsNone(encode(self.row(),self.tokenizer,2,'response'))
        self.assertTrue(inspect_example(self.row(),self.tokenizer,2)['truncated'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.jsonl'
            path.write_text(json.dumps(dict(self.row(),approved_for_training=False))+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'not approved'):
                load_dataset(path,self.tokenizer,512,'response')

    def test_shifted_loss_and_lora_update(self):
        torch.manual_seed(42)
        model=add_lora(self.model(),SimpleNamespace(layers=1,rank=2,alpha=4,four_bit=False))
        row=encode(self.row(),self.tokenizer,512,'response')
        batch={k:torch.tensor([v]) for k,v in row.items()}
        output=model(**batch)
        expected=torch.nn.functional.cross_entropy(output.logits[:,:-1].reshape(-1,output.logits.shape[-1]),
            batch['labels'][:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(output.loss,expected)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        trainable={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        output.loss.backward();optimizer.step()
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen))
        self.assertTrue(any(not torch.equal(p,trainable[n]) for n,p in model.named_parameters() if n in trainable))

    def test_pdf_discretionary_hyphen(self):
        self.assertEqual(clean('eye\ufffe\nopener'),'eyeopener')

    def test_complete_cpu_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/'tiny-base';base.mkdir()
            self.tokenizer.save_pretrained(base)
            self.model().save_pretrained(base)
            for split in ('train','validation'):
                row=dict(self.row(),id=split,split=split,approved_for_training=True,source={'group_id':split})
                (root/f'{split}.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            config=PilotConfig(model=str(base),train=str(root/'train.jsonl'),valid=str(root/'validation.jsonl'),
                output=str(root/'run'),steps=2,max_length=512,max_new_tokens=4,layers=1,rank=2,alpha=4,threads=2)
            manifest, comparisons=run_pilot(config)
            self.assertFalse(manifest['final_test_used'])
            self.assertEqual(manifest['device'],'cpu')
            self.assertTrue(all('before' in r and 'after' in r for r in comparisons))
            self.assertTrue(all(r['before_generation']['generated_tokens']<=4 and
                r['after_generation']['generated_tokens']<=4 for r in comparisons))
            self.assertTrue((root/'run/adapter/adapter_config.json').exists())
            reloaded, tokenizer = load_adapter(root/'run/adapter', None, 'cpu')
            encoded = encode(self.row(), tokenizer, 512, 'response')
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(reloaded(**{k:torch.tensor([v]) for k,v in encoded.items()}).loss))
            self.assertEqual(len((root/'run/loss.csv').read_text().splitlines()),3)
            with self.assertRaisesRegex(ValueError,'empty output'):
                run_pilot(config)


if __name__=='__main__':
    unittest.main()
