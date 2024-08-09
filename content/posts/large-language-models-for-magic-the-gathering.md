---
title: "Large Language Models for Magic: the Gathering"
date: "2024-05-24"
draft: false
---
Magic: The Gathering (MTG) has always fascinated me with its complexity and strategic depth, thanks to its extensive rulebook and vast array of unique cards. Despite the game's popularity, AI systems specifically designed for MTG have been few and far between, often falling short due to their inability to accurately interpret the intricate rules and interactions between cards. This blog post chronicles my recent endeavor to bridge this gap with large language models (LLMs) by creating a specialized dataset and evaluation metric to improve AI performance in MTG-related tasks.

## The Challenge of MTG for AI

MTG presents two primary challenges for players: deck construction and in-game decision-making. With over 27,000 unique cards and a rulebook nearing 300 pages, understanding card interactions and making optimal plays can be daunting. Current AI models often struggle with these aspects, leading to frequent hallucinations and misunderstandings.

## Custom Dataset and MTG-Eval Metric

To address these challenges, I created a custom dataset of MTG-related question-answer pairs, along with an evaluation metric named MTG-Eval. This dataset aims to train and assess language models on their understanding of MTG rules and card interactions. The dataset is divided into three categories:

1. **Card Descriptions**: Questions like "What does card X do?" with answers formatted to mimic card rules text. This helps the model reduce hallucinations by familiarizing it with the text of each card.
   
2. **Rules Questions**: Derived from rulings by the MTG Rules Committee, these questions clarify niche interactions and game scenarios. The official rulings serve as ground-truth answers.

3. **Card Interactions**: Involves questions about combos and card synergies, such as "What is a combo with card X?" or "How can I achieve Y?" The data for this category comes from [Commander Spellbook](https://commanderspellbook.com/), a comprehensive MTG combo database.

## Methodology

### Data Generation

Data from [MTGJSON](https://mtgjson.com/) and Commander Spellbook was used to generate over 80,000 question-answer pairs. The generation process involved using ChatGPT 3.5 to reformat existing data into conversational questions and answers. This synthetic dataset covers a wide range of possible game states and interactions, providing a robust foundation for training.

### Training Process

I fine-tuned Llama 3 8B Instruct, an open-source conversational LLM from Meta, using the custom dataset. The training employed QLoRA to minimize computational requirements, with hyperparameters r=64 and alpha=32 over 75 steps.

## Evaluation

I evaluated the model's performance using a subset of the dataset reserved for testing. Both the base and fine-tuned models were assessed on their ability to answer questions from the card interactions and rules categories, which require deeper comprehension. I used GPT-4 to score the answers on a scale of 1-5, providing consistent and efficient evaluations.

## Results and Impact

The fine-tuned model demonstrated a 10.5% improvement over the base model, with an average score increase from 1.62 to 1.79. This indicates a moderate improvement in the model's understanding of MTG rules and interactions, but both models still have tremendous room to learn.

## Future Directions

Future research could explore the addition of custom tokens for special game symbols like mana and tapping, which are underrepresented in pre-training data and maybe not be appropriately tokenized. Additionally, expanding the dataset to include more diverse game scenarios and interactions could further refine the model's capabilities.

## Conclusion

This project showcases the potential of LLMs to enhance the MTG playing experience and the challenges that still need to be overcome to get there. I hope other people will find this dataset useful for training future models and build off my work. There's certainly much more that could be done and I can't wait for the day when AI systems can build good decks with my janky pet cards.

## Acknowledgments

Thanks to the team at Commander Spellbook for generously sharing their dataset, without which this project would not have been possible. All generated data is unofficial Fan Content permitted under the Fan Content Policy. Not approved/endorsed by Wizards. Portions of the materials used are property of Wizards of the Coast. ©Wizards of the Coast LLC.

For more details and access to the dataset and model, visit the following links:
- [MTG-Eval Dataset](https://huggingface.co/datasets/jakeboggs/MTG-Eval)
- [Fine-Tuned Model](https://huggingface.co/jakeboggs/MTG-Llama)
- [Training Code on GitHub](https://github.com/JakeBoggs/Large-Language-Models-for-Magic-the-Gathering)