from os.path import dirname, realpath, join
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge
# from __init__ import model, tokenizer, rouge

# Lazy loading for ROUGE metrics, model and tokenizer
model = None
tokenizer = None
rouge = None

def summarize(paragraph, gold_summary=None, min_length=60, max_length=100, num_beams=4, verbose=True):

    '''
    [PARAMS]
        paragraph: a string, blank between sentences. Required.
        gold_summary: expected summary content. Optional, used to calculate ROUGE scores.
        min_length, max_length to explicitely indicate length of generated summary. Optional.
        num_beams: used for beam-search in generation. Optional.
        verbose: print info
    [RETURN]
        Generated summary.
        If gold summary provided, also include ROUGE metrics of ROUGE-1,2,L's P,R,F scores. In dict form.
    '''
    # paragraph: a string, blank between sentences. Required.
    # gold_summary: expected summary content. Optional, used to calculate ROUGE scores.
    # min_length, max_length to explicitely indicate length of generated summary. Optional.
    # num_beams: used for beam-search in generation. Optional.

    global model, tokenizer, rouge
    pretrained_path = join(dirname(realpath(__file__)), 'pretrained')
    print("debug:", pretrained_path)
    if model is None:
        if verbose:
            print("Loading model...")
        model = BartForConditionalGeneration.from_pretrained(pretrained_path)
        # model.cuda().eval().half()
        model.eval()
    if verbose:
        print("Model loaded.")

    if tokenizer is None:
        if verbose:
            print("Loading tokenizer...")
        tokenizer = BartTokenizer.from_pretrained(pretrained_path)
    if verbose:
        print("Tokenizer loaded.")

    if rouge is None and gold_summary is not None:
        if verbose:
            print("Loading ROUGE...")
        rouge = Rouge()

    enc_inp = tokenizer.batch_encode_plus([paragraph], max_length=1024, \
        return_tensors='pt')

    if verbose:
        print("Generating sequences...")
    with torch.no_grad():
        # gen_outs = model.generate(enc_inp['input_ids'].cuda(), attention_mask=enc_inp['attention_mask'].cuda(),\
        gen_outs = model.generate(enc_inp['input_ids'], attention_mask=enc_inp['attention_mask'],\
                            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, \
                            min_length=min_length, max_length=max_length, num_beams=num_beams)
    
    # generated summary
    gen_res = tokenizer.decode(gen_outs[0]).strip().replace('<s>','').replace('<pad>','').replace('</s>','')

    if verbose:
        print("Finished generation.")
    if gold_summary is None:
        # no gold is provided, only return generated summary
        return gen_res
    else:    
        # provide ROUGE scores otherwise
        scores = rouge.get_scores(gen_res, gold_summary)
        return gen_res, scores


if __name__ == "__main__":
    para = "Washington (CNN) -- President Obama should end a longstanding policy of not writing letters of condolence to families of troops who commit suicide, dozens of lawmakers urged him in a letter Wednesday. The lawmakers warned that \"our armed forces are in the midst of a suicide epidemic.\" U.S. Army statistics show that more than 200 troops have killed themselves this year, more than in 2008, which was a record year. \"By overturning this policy on letters of condolence to the families of suicide victims, you can send a strong signal that you will not tolerate a culture in our armed forces that discriminates against those with a mental illness,\" the lawmakers wrote. The White House has begun a review of the condolence policy, which the 46 members of Congress said should be completed as soon as possible. They also argued the policy of treating suicides differently from deaths in action flew in the face of military funeral procedure, which treats both the same. Service members who kill themselves get \"a full military burial, complete with flag-draped coffin and 21-gun salute. We have not heard of any reports that military morale and discipline have waned as a result,\" they wrote. They also argued that letters of condolence are \"as much about respect for the personal loss that a family experiences as it is about an acknowledgment by our nation that we have lost a soldier.\" The White House said two weeks ago its review of the policy should \"hopefully\" conclude shortly. White House press secretary Robert Gibbs said the president himself asked for the review. \"If the president didn\'t care, the policy would remain unchanged and unexamined,\" Gibbs said at a December 9 news conference. Despite this year\'s rise in suicides, Army officials said a recent trend downward could signal progress in programs intended to reduce the problem. Between January and mid-November, 140 active-duty soldiers killed themselves, as did 71 Army Reserve and National Guard soldiers. That\'s a total of 211 as of November 17, when Gen. Peter Chiarelli, the Army vice chief of staff, briefed reporters about the problem. But he said the monthly numbers are starting to slow down as the year nears its end. \"This is horrible, and I don\'t want to downplay the significance of these numbers in any way,\" Chiarelli said. For all of 2008, the Army said 140 active-duty soldiers killed themselves, while 57 Guard and Reserve soldiers committed suicide, totaling 197. While the lawmakers cited attitudes toward the mentally ill, the Army is still trying to tackle why soldiers are killing themselves. \"We still haven\'t found any statistically significant causal linkage that would allow us to effectively predict human behavior. The reality is, there is no simple answer -- each suicide case is as unique as the individuals themselves,\" Chiarelli said. He also said there were troubling new statistics showing an increase in suicide rates among young soldiers who have never deployed, another factor puzzling Army researchers. CNN\'s Adam S. Levine, Larry Shaughnessy, Mike Mount and Elaine Quijano contributed to this report."
    gold_sum = "Forty-six members of Congress ask President Obama to revise policy. White House does not send condolence letters to families of suicide victims. Obama asked for review of policy, spokesman says. Policy stigmatizes mental illness, lawmakers say"

    res, scores = summarize(para, gold_sum, num_beams=1)
    print(res)
    print("\n", scores)
    