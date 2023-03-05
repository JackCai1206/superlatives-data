# bingchilling.py

from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

from fake_data import fake_prompts, fake_df
from read_quizzes import get_full_normalized_counts

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # sentence_embeddings = model_output.text_embeds

    # Normalize embeddings
    # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)s

    return sentence_embeddings


if __name__ == '__main__':
    df = get_full_normalized_counts()
    # df = fake_df

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    questions = df.index.tolist()
    prompts = ['apple banana mango']
    question_embeddings = get_sentence_embeddings(questions, tokenizer, model)  # (num_questions, 384)
    prompt_embeddings = get_sentence_embeddings(prompts, tokenizer, model) # (num_prompts, 384)

    # for each person, their overall embedding is the average of their responses weighted by their percentage of votes
    # (num_questions, num_people) * (num_questions, 384) --> (num_people, 384)
    percentages = torch.Tensor(df.values)
    person_embeddings = percentages.T @ question_embeddings
    # normalize the embeddings 
    person_embeddings = F.normalize(person_embeddings, p=2, dim=1)
    # print(percentages.shape, person_embeddings.shape)

    # plot the embeddings and save it to a file
    if False:
        print('making tsne...')
        tsne = TSNE(n_components=2, perplexity=3)
        print('tsne done')
        tsne_coords = tsne.fit_transform(person_embeddings)
        names = df.columns
        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1])
        # add labels to each point
        for i, name in enumerate(names):
            plt.text(tsne_coords[:, 0][i], tsne_coords[:, 1][i], name)
        plt.savefig('embeddings_tsne_p3.png')
    else:
        # new question
        similarity_matrix = prompt_embeddings @ person_embeddings.T
        plt.bar(range(similarity_matrix.shape[1]), height=torch.squeeze(similarity_matrix))
        plt.xticks(range(similarity_matrix.shape[1]), df.columns, rotation=90)
        plt.title(prompts[0])

        # plt.colorbar()
        plt.savefig('new_question.png')
