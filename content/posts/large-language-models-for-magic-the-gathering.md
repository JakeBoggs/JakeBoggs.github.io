---
title: "Large Language Models for Magic: the Gathering"
date: "2024-05-24"
draft: false
---
Context: this was a project from my class CSC 422 - Automated Learning and Data Analysis. I dumped the content from the final paper here with some slight tweaks to make a blogpost.

## Introduction

Magic: The Gathering (MTG) has always fascinated me with its complexity and strategic depth, thanks to its extensive rulebook and vast array of unique cards. Despite the game's popularity, AI systems specifically designed for MTG have been few and far between, often falling short due to their inability to accurately interpret the intricate rules and interactions between cards. This blog post chronicles my recent endeavor to bridge this gap with large language models (LLMs) by creating a specialized dataset and evaluation metric to improve AI performance in MTG-related tasks.

## The Challenge of MTG for AI

MTG presents two primary challenges for players: deck construction and in-game decision-making. With over 27,000 unique cards and a rulebook nearing 300 pages, understanding card interactions and making optimal plays can be daunting. Current AI models often struggle with these aspects, leading to frequent hallucinations and misunderstandings.

## Custom Dataset and MTG-Eval Metric

To address these challenges, I created a custom dataset of MTG-related question-answer pairs, along with an evaluation metric named MTG-Eval. This dataset aims to train and assess language models on their understanding of MTG rules and card interactions. The dataset is divided into three categories:

1.  **Card Descriptions**: Questions like "What does card X do?" with answers formatted to mimic card rules text. This helps the model reduce hallucinations by familiarizing it with the text of each card. The card information was structured programmatically like this:

    ```javascript
    // From generateDescriptions.js
    function createDescription(card) {
        let description = `Name: ${card.name}`;

        if (card.manaCost)
            description += `Mana Cost: ${card.manaCost}`;

        description += `Type: ${card.type}`;

        if (card.power !== undefined)
            description += `Power/Toughness: ${card.power}/${card.toughness}`;

        if (card.loyalty)
            description += `Loyalty: ${card.loyalty}`;

        if (card.text)
            description += `Abilities: ${card.text}`;

        // Replace em dashes with standard hyphens for consistency
        return description.replace(/—/g, '-');
    }
    ```

2.  **Rules Questions**: Derived from rulings by the MTG Rules Committee, these questions clarify niche interactions and game scenarios. The official rulings serve as ground-truth answers. GPT-3.5 was prompted to reformat these rulings into questions:

    ```javascript
    // From generateRulings.js - Inside the generation loop
    const completion = await openai.chat.completions.create({
        messages: [
            {
                role: 'system',
                content: 'You are a helpful assistant designed to output JSON.',
            },
            {
                role: 'user',
                // Dynamically insert card description and ruling
                content: `Below is a Magic: the Gathering card and an official ruling associated with it. Reformat the ruling into a simple question.\n${createDescription(cards[rulings[ruling].name][0])}\nRuling: ${rulings[ruling].ruling}\nRespond in the following JSON format: {"question": "INSERT QUESTION HERE"}`
            }
        ],
        model: 'gpt-3.5-turbo-0125',
        response_format: { type: 'json_object' }
    });

    data.push({
        instruction: JSON.parse(completion.choices[0].message.content).question,
        response: ruling, // The original ruling text serves as the answer
        category: 'ruling'
    });
    ```

3.  **Card Interactions**: Involves questions about combos and card synergies, such as "What is a combo with card X?" or "How can I achieve Y?" The data for this category comes from [Commander Spellbook](https://commanderspellbook.com/), a comprehensive MTG combo database. The prompt guided the LLM to create conversational questions and detailed answers based on the combo data:

    ```javascript
    // From generateInteractions.js - Inside the generation loop
    const completion = await openai.chat.completions.create({
        messages: [
            {
                role: 'system',
                content: 'You are a helpful assistant designed to output JSON.',
            },
            {
                role: 'user',
                // Provide combo details to the LLM
                content: `I will give you the steps of Magic: the Gathering combo and you will create a question asking about a combo that can be performed with one of the required cards, along with the corresponding answer. You don't need to specify that the combo is about Magic in the question. When answering, use conversational language and first explain what other cards are required, then describe all of the steps needed to perform the combo, as well as the results. Be concise, but don't leave out any steps. Respond in JSON in the following format: {"question": "[QUESTION]", "answer": "[ANSWER]"}\nCards required: ${combo.uses.map(x => x.card.name).join(', ')}\n${combo.otherPrerequisites.length > 0 ? 'Prerequisites: ' + combo.otherPrerequisites + '\n': ''}Combo steps:\n${combo.description}\nResults:\n${combo.produces.map(x => x.name).join(', ')}`
            }
        ],
        model: 'gpt-3.5-turbo-0125',
        response_format: { type: 'json_object' }
    });

    data.push({
        instruction: JSON.parse(completion.choices[0].message.content).question,
        response: JSON.parse(completion.choices[0].message.content).answer,
        category: 'combo'
    });
    ```

## Methodology

### Data Generation

Data from [MTGJSON](https://mtgjson.com/) and Commander Spellbook was used to generate over 80,000 question-answer pairs. The generation process involved using ChatGPT 3.5 to reformat existing data into conversational questions and answers, as shown in the snippets above. This synthetic dataset covers a wide range of possible game states and interactions, providing a robust foundation for training.

### Training Process

I fine-tuned Llama 3 8B Instruct, an open-source conversational LLM from Meta, using the custom dataset. The training employed QLoRA (Quantized Low-Rank Adaptation) to minimize computational requirements. Here's the configuration used:

```python
# From train.py - QLoRA and PEFT configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target specific layers for LoRA adaptation
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
```

The dataset was formatted into the required chat structure using the Hugging Face `transformers` library:

```python
# From train.py - Formatting data for chat model
def format_chat_template(example):
    # Structure the data into user/assistant turns
    convos = [{
        'role': 'user',
        'content': example['instruction']
    },
    {
        'role': 'assistant',
        'content': example['response']
    }]
    # Apply the tokenizer's chat template
    texts = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
    return { 'text' : texts }

dataset = dataset.map(format_chat_template)
```

The model was trained using the `SFTTrainer` from the `trl` library with the following arguments over 75 steps:

```python
# From train.py - SFTTrainer setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field = 'text', # Field containing formatted chat data
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False, # Important for chat format
    peft_config=peft_config,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 75, # Relatively short fine-tuning run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = 'adamw_8bit',
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        output_dir='outputs'
    )
)
trainer.train()
trainer.save_model(new_model)
```

## Evaluation

I evaluated the model's performance using a subset of the dataset reserved for testing. Both the base and fine-tuned models were assessed on their ability to answer questions from the card interactions and rules categories, which require deeper comprehension.

First, the model generated an answer for a given instruction:

```python
# From evaluate.py - Generating model response
prompt = tokenizer.apply_chat_template([{
    'role': 'user',
    'content': example['instruction']
}], tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to('cuda')

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])

decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
student_answer = decoded.split("assistant")[1] # Extract the generated answer
```

Then, I used GPT-4 to score the answers on a scale of 1-5, providing consistent and efficient evaluations. GPT-4 was prompted with the original question, the ground-truth answer, and the model's generated answer:

```python
# From evaluate.py - Payload for GPT-4 evaluation
payload = {
    "model": "gpt-4-turbo",
    "response_format": { "type": 'json_object' },
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    # The prompt instructing GPT-4 how to grade
                    "text": "You are an expert grader who will be given a Magic: the Gathering related question and correct answer, along with a students response. You rate the student's response on a scale of 1-5, with 5 being entirely correct and 1 being entirely incorrect. Accurately identify any issues with the answer and explain why the students response is entirely correct, partially correct, or entirely incorrect. Reply in the following JSON format: {\"explanation\": \"[EXPLANATION]\", \"score\": SCORE}\n\"Question: "+ example['instruction'] + "\nCorrect answer: "+ example['response'] + "\nStudent's answer: " + student_answer
                }
            ]
        }
    ]
}

response = json.loads(requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content'])
score = float(response['score'])
```

## Results

The fine-tuned model demonstrated a 10.5% improvement over the base model, with an average score increase from 1.62 to 1.79 based on the GPT-4 evaluation. This indicates a slight improvement in the model's understanding of MTG rules and interactions, but both models still have tremendous room to learn.

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