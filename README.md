<h1 align="center"><img src="img/logo.png" alt="Galois Autocompleter"/><p>Galois Autocompleter</p></h1>

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://twitter.com/iedmrc">
    <img alt="Twitter: iedmrc" src="https://img.shields.io/twitter/follow/iedmrc.svg?style=social" target="_blank" />
  </a>
</p>

> An autocompleter server for code editors based on [OpenAI GPT-2](https://github.com/openai/gpt-2).

**Galois Autocompleter** is a **Deep Learning Code Autocompleter** based on [OpenAI GPT-2](https://github.com/openai/gpt-2). Currently we provide a Java Model trained (fine-tuned) on a curated list of approximately 8GB of Java source codes gathered from the Github repositories with number of stars >= 1500.

This project is based on [Cortex](https://docs.cortex.dev/) and [HuggingfFace Transformers](https://huggingface.co/transformers/model_doc/gpt2.html).

![Galois Demo](https://user-images.githubusercontent.com/30511610/98991778-11e4b000-250b-11eb-8905-527bf9a4f203.png)

## Installation
With a Python 3.6 environment, you only need to install [Cortex](https://docs.cortex.dev/) by the following command:
```sh
pip install cortex
```

## Running
On the root folder of Galois repository, run the following command to deploy the server:
```sh
cortex deploy
```

Checking the logs: 
```sh
cortex logs galois-autocompleter
```

## Usage
Currently, there are Plugins for both [IntelliJ IDEA](https://github.com/GabrielTamujo/galois-plugin-intellij) and [VS Code](https://github.com/GabrielTamujo/galois-plugin-vscode). The server expects the request body as the following example:

```sh
{
    text: "public class City {",
}

```
You can even add optional keys setting some [generation parameters](https://huggingface.co/transformers/main_classes/model.html?#transformers.generation_utils.GenerationMixin.generate):

```sh
{
    text: "public class City {",
    top_p: 0.85, 
    top_k: 50, 
    temperature: 0.5 
}
```

An example curl command:

```sh
curl -X POST \
  http://localhost:8889/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"public class City"}'
  ```

## Planned Works
- New models for Javascript, Python and any possible language. 
- Create extensions for more editors.

## Contribution
Contributions are welcome desired. Feel free to create an issue or a pull request.

## License
It is licensed under MIT License as found in the LICENSE file.

## Disclaimer
This repo has no affiliation or relationship with OpenAI.