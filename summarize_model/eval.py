import pickle
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge


def main(save_pretrained=False):
    rouge = Rouge()

    # Quick reload
    with open('./data/all_toks.dat', 'rb') as f:
        t_toks, v_toks = pickle.load(f)

    if save_pretrained:
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

        tokenizer.save_pretrained('./pretrained')
        model.save_pretrained('./pretrained')
    
    else:
        tokenizer = BartTokenizer.from_pretrained('./pretrained')
        model = BartForConditionalGeneration.from_pretrained('./pretrained')

    model.cuda().eval()
    model.half()

    stats = {}
    stats['rouge-1'] = {}
    stats['rouge-1']['f'] = 0
    stats['rouge-1']['p'] = 0
    stats['rouge-1']['r'] = 0
    stats['rouge-2'] = {}
    stats['rouge-2']['f'] = 0
    stats['rouge-2']['p'] = 0
    stats['rouge-2']['r'] = 0
    stats['rouge-l'] = {}
    stats['rouge-l']['f'] = 0
    stats['rouge-l']['p'] = 0
    stats['rouge-l']['r'] = 0
    
    i = 0
    already = 0
    gen_out = 0
    batch_sz = 8
    while i<len(v_toks):
        
        cur_paras, cur_sums = [],[]
        for j in range(batch_sz):
            if (i)>=len(v_toks):
                break
            cur_para = v_toks[i][0]
            cur_para = tokenizer.decode(cur_para).replace("<s>"," ").replace("</s>"," ")
            cur_sum = v_toks[i][1]
            cur_sum = tokenizer.decode(cur_sum).replace("<s>"," ").replace("</s>"," ")
            
            cur_paras.append(cur_para)
            cur_sums.append(cur_sum)
            i += 1
            
            
        enc_inp = tokenizer.batch_encode_plus(cur_paras, max_length=1024, return_tensors='pt', pad_to_max_length=True)
        #dec_inp = tokenizer.batch_encode_plus([cur_sum], max_length=1024, return_tensors='pt')
        
        with torch.no_grad():
            gen_outs = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=60, max_length=100, num_beams=4)
        
        for gen_out,cur_sum in zip(gen_outs,cur_sums):
            gen_res = tokenizer.decode(gen_out).replace('<s>','').replace('<pad>','')
            score = rouge.get_scores(gen_res, cur_sum)[0]

            for met in stats:               
                for it in stats[met]:
                    stats[met][it] += score[met][it]
            already += 1
        
        #print("\r",end='',flush=True)
        sc1 = str(stats['rouge-1']['r']/(already))[:7]
        sc2 = str(stats['rouge-2']['r']/(already))[:7]
        sc3 = str(stats['rouge-l']['r']/(already))[:7]
        perc = str(100*(i+1)/len(v_toks))[:4]
        
        print("\rROUGE-1: %s. ROUGE-2: %s. ROUGE-L: %s. %s%%."%(sc1,sc2,sc3,perc),\
            end='',flush=True)

        if i>=600:
            break


if __name__ == "__main__":
    main()