import pickle
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge


def generate_post_dataset():

    tokenizer = BartTokenizer.from_pretrained('./pretrained')
    model = BartForConditionalGeneration.from_pretrained('./pretrained')

    model.cuda().eval()
    model.half()

    file_path = "D:/learn/第六学期/CompNet_Project/lately-export-ContentPost/"
    for file_id in range(1,4):
        
        cur_res = []
        cur_batch_res = []
        file_name = "ContentPost."+str(file_id)+".json"
        save_file_name = "Result_"+str(file_id)+".json"

        with open(file_path+file_name, "r", encoding='utf-8') as f:
            dat = json.load(f)

        i = 0
        gen_out = 0
        batch_sz = 32
        while i<len(dat):

            cur_batch_res = []
            cur_paras, cur_sums = [],[]
            for j in range(batch_sz):
                if (i)>=len(dat):
                    break

                cur_dat = dict()
                cur_dat['content'] = dat[i]['content']
                cur_batch_res.append(cur_dat)
                cur_para = cur_dat['content']
                cur_paras.append(cur_para)
                i += 1

                if i%32==0:
                    print("\rFile No. %d. Completed: %s%%"%(file_id,str(100*(i)/len(dat))),end='',flush=True)
                
                
            enc_inp = tokenizer.batch_encode_plus(cur_paras, max_length=256, return_tensors='pt', pad_to_max_length=True)
            #dec_inp = tokenizer.batch_encode_plus([cur_sum], max_length=1024, return_tensors='pt')
            
            with torch.no_grad():
                gen_outs = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                    bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=20, max_length=50, num_beams=8)
            
            for j, gen_out in enumerate(gen_outs):
                gen_res = tokenizer.decode(gen_out).replace('<s>','').replace('<pad>','').replace('</s>','')
                cur_batch_res[j]['generated_summary'] = gen_res
            
            cur_res += cur_batch_res

            with open(file_path+save_file_name, "w", encoding='utf-8') as f:
                json.dump(cur_res, f, indent=4)
            
            #if i>=24:
            #    break


def generate_baiwei_dataset():

    tokenizer = BartTokenizer.from_pretrained('./pretrained')
    model = BartForConditionalGeneration.from_pretrained('./pretrained')

    model.cuda().eval()
    model.half()

    file_name = r"D:\learn\第六学期\CompNet_Project\baiwei\abi_press.json"
    save_file_name = r"D:\learn\第六学期\CompNet_Project\baiwei\abi_press_summary.json"
        
    cur_res = []
    cur_batch_res = [] 

    with open(file_name, "r", encoding='utf-8') as f:
        dat = json.load(f)

    i = 0
    gen_out = 0
    batch_sz = 4
    while i<len(dat):

        cur_batch_res = []
        cur_paras, cur_sums = [],[]
        for j in range(batch_sz):
            if (i)>=len(dat):
                break

            cur_dat = dict()
            cur_dat['title'] = dat[i]['title']
            cur_dat['category'] = dat[i]['category']
            cur_dat['article_body'] = dat[i]['article_body']

            cur_batch_res.append(cur_dat)
            cur_para = dat[i]['title']+'. '+dat[i]['article_body']
            cur_paras.append(cur_para)
            i += 1

            if i%4==0:
                print("\rCompleted: %s%%"%(str(100*(i)/len(dat))),end='',flush=True)
            
            
        enc_inp = tokenizer.batch_encode_plus(cur_paras, max_length=1024, return_tensors='pt', pad_to_max_length=True)
        
        with torch.no_grad():
            gen_outs_1_beam = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=60, max_length=100, num_beams=1)
            gen_outs_8_beams = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=60, max_length=100, num_beams=8)
        
        for j, (gen_out_1, gen_out_8) in enumerate(zip(gen_outs_1_beam, gen_outs_8_beams)):
            gen_res = tokenizer.decode(gen_out_1).replace('<s>','').replace('<pad>','').replace('</s>','')
            cur_batch_res[j]['generated_summary_1_beam'] = gen_res
            gen_res = tokenizer.decode(gen_out_8).replace('<s>','').replace('<pad>','').replace('</s>','')
            cur_batch_res[j]['generated_summary_8_beams'] = gen_res
        
        cur_res += cur_batch_res

        with open(save_file_name, "w", encoding='utf-8') as f:
            json.dump(cur_res, f, indent=4, ensure_ascii=False)
        
        #if i>=24:
        #    break



def generate_blog_dataset():

    tokenizer = BartTokenizer.from_pretrained('./pretrained')
    model = BartForConditionalGeneration.from_pretrained('./pretrained')

    model.cuda().eval()
    model.half()

    file_name = r"D:\learn\第六学期\CompNet_Project\blogs\lately_blogs.json"
    save_file_name = r"D:\learn\第六学期\CompNet_Project\blogs\lately_blogs_summary.json"
        
    cur_res = []
    cur_batch_res = [] 

    with open(file_name, "r", encoding='utf-8') as f:
        dat = json.load(f)

    i = 0
    gen_out = 0
    batch_sz = 4
    while i<len(dat):

        cur_batch_res = []
        cur_paras, cur_sums = [],[]
        for j in range(batch_sz):
            if (i)>=len(dat):
                break

            cur_dat = dict()
            cur_dat['Title'] = dat[i]['Title']
            cur_dat['Content'] = dat[i]['Content']
            cur_dat['Permalink'] = dat[i]['Permalink']

            cur_batch_res.append(cur_dat)
            cur_para = dat[i]['Title']+'. '+dat[i]['Content']
            cur_paras.append(cur_para)
            i += 1

            if i%4==0:
                print("\rCompleted: %s%%"%(str(100*(i)/len(dat))),end='',flush=True)
            
            
        enc_inp = tokenizer.batch_encode_plus(cur_paras, max_length=1024, return_tensors='pt', pad_to_max_length=True)
        
        with torch.no_grad():
            gen_outs_1_beam = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=60, max_length=100, num_beams=1)
            gen_outs_8_beams = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
                                bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, min_length=60, max_length=100, num_beams=8)
        
        for j, (gen_out_1, gen_out_8) in enumerate(zip(gen_outs_1_beam, gen_outs_8_beams)):
            gen_res = tokenizer.decode(gen_out_1).replace('<s>','').replace('<pad>','').replace('</s>','')
            cur_batch_res[j]['Summary_1_beam'] = gen_res
            gen_res = tokenizer.decode(gen_out_8).replace('<s>','').replace('<pad>','').replace('</s>','')
            cur_batch_res[j]['Summary_8_beams'] = gen_res
        
        cur_res += cur_batch_res

        with open(save_file_name, "w", encoding='utf-8') as f:
            json.dump(cur_res, f, indent=4, ensure_ascii=False)
        
        #if i>=24:
        #    break



def main(save_pretrained=False):
    rouge = Rouge()

    # Quick reload
    with open(r'D:\learn\第六学期\CompNet_Project\all_toks.dat', 'rb') as f:
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
            gen_res = tokenizer.decode(gen_out).replace('<s>','').replace('<pad>','').replace('</s>','')
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
    generate_blog_dataset()
    #generate_baiwei_dataset()
    #generate_post_dataset()
    #main()