# Prompt Engineering Techniques

Prompt engineering is the discipline of crafting inputs to large language models to elicit desired outputs. As LLMs have grown in capability, sophisticated prompting strategies have emerged that significantly improve reasoning, accuracy, and task performance.

## Foundational Techniques

### Zero-Shot Prompting

Zero-shot prompting provides the model with a task description and no examples. The model relies entirely on its pre-trained knowledge. For instance, asking "Classify the sentiment of this review: 'The food was amazing'" expects the model to understand sentiment classification without demonstrations.

### Few-Shot Prompting

Few-shot prompting includes a small number of input-output examples before the actual query. These demonstrations establish the expected format and reasoning pattern. Research shows that the choice, ordering, and format of examples significantly impact performance — sometimes more than the number of examples provided.

## Advanced Reasoning Techniques

### Chain of Thought (CoT)

Chain of Thought prompting instructs the model to show its reasoning step by step before arriving at an answer. Adding "Let's think step by step" to a prompt can dramatically improve performance on arithmetic, commonsense reasoning, and symbolic manipulation tasks. CoT works because it decomposes complex reasoning into intermediate steps the model can handle individually.

**Zero-shot CoT** simply appends "Let's think step by step" without examples. **Few-shot CoT** provides examples that include explicit reasoning chains.

### Self-Consistency

Self-consistency improves CoT by sampling multiple reasoning paths and selecting the most common final answer through majority voting. Instead of relying on a single greedy decoding path, it leverages the intuition that correct reasoning tends to converge on the same answer even through different intermediate steps.

### Tree of Thoughts (ToT)

Tree of Thoughts generalizes CoT by allowing the model to explore multiple reasoning branches at each step. It uses breadth-first or depth-first search through the space of possible reasoning chains, with the model itself evaluating which branches are most promising. This enables deliberate planning and backtracking.

### ReAct Prompting

ReAct combines reasoning and acting in an interleaved fashion. The model alternates between generating reasoning traces ("I need to search for X because...") and taking actions (calling a search API). This grounds the model's reasoning in external information, reducing hallucination.

## Structural Techniques

### Retrieval-Augmented Generation (RAG)

RAG augments prompts with relevant external knowledge retrieved from a document store. The retrieval step uses semantic similarity (usually embedding-based) to find pertinent context, which is then injected into the prompt. This technique is especially effective for knowledge-intensive tasks where the model's parametric memory is insufficient or outdated.

### Automatic Prompt Engineering (APE)

APE uses LLMs to generate and optimize prompts automatically. The system proposes candidate prompts, evaluates them on a validation set, and iteratively refines the best performers. This removes human guesswork from the prompt design process.

## Best Practices

1. **Be specific**: Vague instructions produce vague outputs. Specify format, length, style, and constraints.
2. **Provide context**: Include relevant background information the model needs.
3. **Use delimiters**: Clearly separate instructions, context, and input using markers like triple quotes or XML tags.
4. **Request structured output**: Ask for JSON, markdown, or other formats when you need parseable results.
5. **Iterate systematically**: Change one variable at a time to understand what drives improvements.

## Evaluation

Prompt effectiveness should be measured quantitatively. Common metrics include task accuracy, F1 score for classification, BLEU/ROUGE for generation, and human preference ratings. A/B testing different prompt variants on representative datasets is essential before deploying to production.
