import numpy as np
import argparse

def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Answer:' == line[:len('Answer:')]:
                answer = line.replace('Answer:', '').strip().lower()
                answers.append(answer)
                # print(answer)
            if 'LLM:' == line[:len('LLM:')]:
                llm_prediction = line.replace('LLM: ', '').strip().lower()
                
                llm_predictions.append(llm_prediction)
                
    return answers, llm_predictions

def evaluate(answers, llm_predictions, k=1):
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    print(predict_num)
    no_empty = 0
    for answer, prediction in zip(answers, llm_predictions):
        if k > 1:
            rank = prediction.index(answer)
            if rank < k:
                NDCG += 1 / np.log2(rank + 1)
                HT += 1
        elif k == 1:
            if prediction[1:] in answer:
                NDCG += 1
                HT += 1
            # if (prediction[1:12] in answer) and (not 'empty title' in answer) :
            #     NDCG += 1
            #     HT += 1
            #     no_empty +=1
                
    return NDCG / predict_num, HT / predict_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='Industrial_and_Scientific')
    parser.add_argument("--save_dir", type=str, default='tallrec')
    args = parser.parse_args()
    
    data = args.data
    model = args.model

    inferenced_file_path = f'./{model}best/{data}_recommendation_output.txt'
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    print(len(answers), len(llm_predictions))
    assert(len(answers) == len(llm_predictions))
    
    ndcg, ht = evaluate(answers, llm_predictions, k=1)
    print(f"Original ndcg at 1: {ndcg}")
    print(f"Original hit at 1: {ht}")
