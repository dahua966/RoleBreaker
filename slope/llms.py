import torch
import datetime
from openai import OpenAI
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from .settings import *

if os.name == "posix":
    from vllm import LLM as vllm
    from vllm import SamplingParams


class LLM(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.model_name = None
        self.device = "cuda"
        self.conversation = []
        self.tools = {}
        self.prompts = []
        self.convs = []

    def add_conv_node(self, conv: Union[Dict[str, str], List[Dict[str, str]]]):
        """Add a node to conversation list --> generate()"""
        if isinstance(conv, dict):
            self.conversation.append(conv)
        elif isinstance(conv, list):
            self.conversation.extend(conv)

    def clear_conv(self):
        self.conversation = []

    def set_conv(self, conv: List[Dict[str, str]]):
        """Set conversation for single generation --> generate()"""
        self.conversation = conv

    def set_tools(self, tools: List[Dict]):
        self.tools = tools

    def add_raw_prompt(self, prompt: str):
        """Add prompt for batch generation --> generate_batch()"""
        self.prompts.append(prompt)

    def add_conv_prompt(self, conversation: List[Dict[str, str]], thinking=False):
        """Add a completed conversation, turn into prompt str --> local model generate_batch()"""
        self.prompts.append(self.conv2prompt(conversation, thinking=thinking))

    def add_conv(self, conversation: List[Dict[str, str]]):
        """Add a completed conversation --> remote model generate_batch()"""
        self.convs.append(conversation)

    def set_prompts(self, prompts):
        self.prompts = prompts

    def clear_prompts(self):
        self.prompts = []

    def generate(self):
        pass

    def generate_batch(self):
        pass

    @staticmethod
    def sys_prompt(prompt):
        return [{"role": "system", "content": prompt}]

    @staticmethod
    def user_prompt(prompt):
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def assistant_prompt(prompt):
        return [{"role": "assistant", "content": prompt}]

    @staticmethod
    def conv_prompt(prompt, response):
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

    def conv2prompt(self, conversation: List[Dict], verbose=False, thinking=False) -> str:
        """将对话列表转化为prompt字符串 考虑模型没有自带对话模板的情况"""
        if self.tokenizer.chat_template is None:
            from fastchat.model import get_conversation_template
            model_name = self.model_name.lower()
            if "llama-3.1" in model_name:
                model_name = "llama-3-"  # patch for fastchat
            elif "glm-4" in model_name:
                model_name = "chatglm3"
            conv_template = get_conversation_template(model_name)
            if conv_template.name == "one_shot":
                cprint(f"[!]Warning: Conversation template is {conv_template.name}!", "yellow")
            cprint(f"\n[*]Conversation template is {conv_template.name}", "blue") if verbose else None

            for msg in conversation:
                if msg["role"] == "system":
                    conv_template.set_system_message(msg["content"])
                elif msg["role"] == "user":
                    conv_template.append_message(conv_template.roles[0], msg["content"])
                elif msg["role"] == "assistant":
                    conv_template.append_message(conv_template.roles[1], msg["content"])

            if conv_template.messages[-1][0] == conv_template.roles[0]:
                conv_template.append_message(conv_template.roles[1], "")
            else:
                raise Exception(f"Conversation's last role is not user.\n{conv_template.messages}\n")

            prompt = conv_template.get_prompt()
        else:
            if not thinking and self.model_name.lower().startswith("qwen3"):
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            # if isinstance(prompt, list) and len(prompt):
            #     prompt = prompt[0]

        if verbose:
            print(f"{colored('Structured prompt:', 'blue', attrs=['bold'])}:\n{prompt}\n")

        return prompt

    def conv2prompt_w_tools(self, conversation: List[Dict], tools, verbose=False) -> str:
        if re.findall("qwen", self.model_name.lower()):
            prompt = self.tokenizer.apply_chat_template(conversation, tools=tools, tokenize=False,
                                                        add_generation_prompt=True)

        if verbose:
            print(f"{colored('Structure prompt:', 'blue', attrs=['bold'])}:\n{prompt}\n")

        return prompt


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='auto',
                 dtype=torch.float16,
                 ):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_path.split(os.sep)[-1]
        self.device = device
        # cprint(f"[*]Loading LLM model: {self.model_name} @ {self.device}\n", "blue")
        s_time = time.time()
        self.model, self.tokenizer = self.create_model(
            dtype,
        )
        cprint(f"[*]Loading LLM model({self.model_name} @ {self.device}) in {time.time() - s_time:.2f} seconds\n",
               "green")

    # @torch.inference_mode()
    def create_model(self, dtype=torch.float16):
        is_higher_than_ampere = torch.cuda.is_bf16_supported()
        try:
            import flash_attn
            is_flash_attn_available = True
        except:
            is_flash_attn_available = False

        # Even flash_attn is installed, if <= Ampere, flash_attn will not work
        if is_higher_than_ampere and is_flash_attn_available:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,  # NumPy doesn't support BF16
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            cprint("[*]Using FP16 and Flash-Attention 2\n", "green")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            cprint("[*]Using FP16 and normal attention implementation", "green")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if model.generation_config.pad_token_id is None:
            cprint(
                f"[*]model.generation_config.pad_token_id is None, set to tokenizer.pad_token_id({tokenizer.pad_token_id})",
                "yellow")
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        fix_tokenizer_pad(tokenizer, self.model_name)

        fix_tokenizer_template(self.model_name, tokenizer)

        return model, tokenizer

    def generate(self, temperature=1.0,
                 max_tokens=512,
                 verbose=False,
                 thinking=False, **kwargs) -> str:
        """Generate from conversation"""
        if self.conversation:
            prompt = self.conv2prompt(self.conversation, verbose, thinking=thinking)
        elif self.prompts:
            prompt = self.prompts[0]
        else:
            cprint(f"[-]Empty conversation or prompt, return None", "red")
            return None

        print(f"[*]Generate response from prompt:\n{prompt}\n") if verbose else None

        if isinstance(prompt[0], str):
            inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        else:
            cprint(f"[!]Prompts is not str, return None", "red")
            return None

        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens, **kwargs
            )  # repetition_penalty=1.0
        else:
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_tokens, **kwargs
            )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0, len(inputs['input_ids'][0]):]
        self.clear_prompts()

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    def generate_batch(self, temperature=1.0,
                       max_tokens=512, **kwargs) -> List[str]:
        """Generate from prompts"""
        if len(self.prompts) == 0:
            cprint(f"[!]LLM batch generating, self.prompts is [], return []", "yellow")
            return []

        if isinstance(self.prompts[0], str):
            inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(self.model.device)
        else:
            print(f"[!]Prompts is not str, return None")
            return None

        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                **kwargs
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_tokens,
                **kwargs
            )

        # print(f"[*]Generate Batch\nModel name: {self.model_name}\nPrompts{self.prompts}\nRaw output_ids:\n{output_ids}\n")
        if self.model_name == "HarmBench-Llama-2-13b-cls":
            output_ids = output_ids[..., -1]  # for harmbench
        elif not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, len(inputs['input_ids'][0]):]

        self.clear_prompts()

        return self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )


class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 device="auto",
                 gpu_memory_utilization=0.9,
                 tensor_parallel_size=1):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        s_time = time.time()
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization, device=device,
            trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        cprint(f"[*]Loading vLLM model ({self.model_name} @ {device}) in {time.time() - s_time:.2f} seconds\n", "green")

    def generate(self, temperature=1.0, max_tokens=512, allowed_token_ids=None, verbose=False, thinking=True) -> str:
        if verbose:
            cprint("Generating from conversation:", 'blue', attrs=['bold'])
            print(self.conversation)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         allowed_token_ids=allowed_token_ids)
        if self.tools:
            prompt = self.conv2prompt_w_tools(self.conversation, self.tools, verbose)
        else:
            if self.conversation:
                prompt = self.conv2prompt(self.conversation, verbose, thinking=thinking)
            elif self.prompts:
                prompt = self.prompts[0]
            else:
                cprint(f"[-]Empty conversation or prompts\n", "red")
                return None
        # print(f"[*]Structured prompt {self.model_name}:\n{prompt}")
        results = self.model.generate(prompt, sampling_params, use_tqdm=False)

        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)

        assert len(outputs) == 1
        response = outputs[0]

        if self.tools:
            tool_res = re.findall(r"<tool_call>([\s\S]*)</tool_call>", response)
            if not tool_res:
                cprint(f"[-]Error results from tool call.\n", "red")
                print(f"[*]Raw response:\n{response}\n")
                return json.dumps({"error": response})
            else:
                tool_res = json.loads(tool_res[0])
                name = tool_res["name"]
                response = json.dumps(tool_res["arguments"])

        return response

    def generate_batch(self, temperature=1.0, max_tokens=512) -> List[str]:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(self.prompts, sampling_params, use_tqdm=False)

        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)

        assert len(outputs) == len(self.prompts)
        self.clear_prompts()
        return outputs


class OpenAILLM(LLM):
    def __init__(self, model_name, base_url, key, repeat_num=5):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.key = key
        self.repeat_num = repeat_num
        self.client = OpenAI(base_url=self.base_url, api_key=self.key)

    def generate(self, **kwargs) -> str:
        for msg in self.conversation:
            if not msg["role"] in ["system", "user", "assistant"]:
                raise Exception(f"[-]Error role in conversation: \n{self.conversation}")

        response = "[!]Error:"
        for i in range(1, self.repeat_num + 1):
            try:
                # if self.tools:
                #     chat = self.client.beta.assistants.create(
                #           instructions="You are a weather bot. Use the provided functions to answer questions.",
                #           model=self.model_name,
                #           tools=self.tools
                #     )
                if self.tools:
                    kwargs["tools"] = self.tools
                    kwargs["tool_choice"] = {"type": "function",
                                             "function": {"name": self.tools[0]["function"]["name"]}}

                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation,
                    timeout=90,
                    **kwargs
                )

                if chat.choices:
                    if chat.choices[0].message.tool_calls:
                        response = chat.choices[0].message.tool_calls[0].function.arguments
                    else:
                        response = chat.choices[0].message.content
                    if response is not None:
                        break
                elif chat.error:
                    eMsg = chat.error.message
                    raise Exception(eMsg)
                else:
                    print("[!]Error and Choice are both None")
                    print(f"[*]Chat:\n{chat}\n")

            except KeyboardInterrupt:
                print("[-]KeyboardInterrupt, exiting...")
                exit()

            except Exception as e:
                err_msg = str(e)[:500]
                print(
                    f"[!]在第{i}次请求 `{self.model_name}` 时报错\n"
                    f"[-]报错信息 ({datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}): \n{err_msg}\n"
                    f"[*]Sleep {10*i}s")
                # print_exc()
                if i < self.repeat_num:
                    time.sleep(10*i)
                else:
                    response += f"\n{err_msg}"
        return response

    def generate_batch(self, threads=16, **kwargs):
        """批量请求 convs，并确保返回结果顺序与输入顺序一致"""

        def resp_conv(idx, conv):
            self.set_conv(conv)
            response = self.generate(**kwargs)
            return idx, response

        results = [None] * len(self.convs)  # 初始化与 self.convs 等长的结果列表

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(resp_conv, idx, conv) for idx, conv in enumerate(self.convs)]
            for future in tqdm(as_completed(futures), total=len(self.convs),
                               desc=f"Batch request {self.model_name}"):
                try:
                    idx, response = future.result()
                    results[idx] = response
                except Exception as exc:
                    print(f'[-]{future} generated an exception: {exc}')

        # assert None not in results, "Some responses are missing"
        self.convs = []
        return results


@Timer(text="[*]加载模型用时: {:.2f}s")
def load_model_and_tokenizer(model_path, device="auto"):
    model_name = model_path.split(os.sep)[-1]
    is_higher_than_ampere = torch.cuda.is_bf16_supported()
    try:
        import flash_attn

        is_flash_attn_available = True
    except:
        is_flash_attn_available = False

    # Even flash_attn is installed, if <= Ampere, flash_attn will not work
    if is_higher_than_ampere and is_flash_attn_available:
        print("[*]Using FP16 and Flash-Attention 2...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            # low_cpu_mem_usage=True,
            torch_dtype=torch.float16,  # NumPy doesn't support BF16
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    else:
        print("[*]Using FP16 and normal attention implementation...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            # low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.clean_up_tokenization_spaces == True:
        print(
            "WARNING: tokenizer.clean_up_tokenization_spaces is by default set to True. "
            "This will break the attack when validating re-tokenization invariance. Setting it to False..."
        )
        tokenizer.clean_up_tokenization_spaces = False

    # 更新模型的 pad_token_id
    if model.config.pad_token_id is None:
        cprint(f"[*]model.config.pad_token_id is None, set to {tokenizer.eos_token_id}", "yellow")
        model.config.pad_token_id = tokenizer.eos_token_id

    fix_tokenizer_pad(tokenizer, model_name)

    fix_tokenizer_template(model_path, tokenizer)

    return model, tokenizer


def fix_tokenizer_pad(tokenizer, model_name):
    """设置填充参数"""
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"
        print(f"[*]tokenizer.padding_side is None, set to left\n")
    else:
        print(f"[*]tokenizer.padding_side default: {tokenizer.padding_side}")
        if tokenizer.padding_side != "left" and "qwen" in model_name.lower():
            cprint(f"[!]Force set tokenizer.padding_side to left.\n", "yellow")
            tokenizer.padding_side = "left"

    # cprint(f"[*]tokenizer.pad_token_id is {tokenizer.pad_token_id}\n", "green")
    # cprint(f"[*]tokenizer.eos_token_id is {tokenizer.eos_token_id}\n", "green")

    if tokenizer.pad_token is None:
        cprint(f"[*]tokenizer.pad_token is None, set to eos_token ({tokenizer.eos_token})\n", "yellow")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        cprint(f"[*]tokenizer.pad_token_id is None, set to eos_token_id ({tokenizer.eos_token_id})\n", "yellow")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def fix_tokenizer_template(model_path, tokenizer):
    """手动添加对话模板"""
    if not tokenizer.chat_template:
        cprint(f"[*]The tokenizer does not come with a chat template.\n", "yellow")

        if "vicuna-7b-v1.5" in model_path.lower():
            tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
                {% set loop_messages = messages[1:] %}
                {% set system_message = messages[0]['content'].strip() + '' %}
            {% else %}
                {% set loop_messages = messages %}
                {% set system_message = '' %}
            {% endif %}
            {{ bos_token + system_message }}
            {% for message in loop_messages %}
                {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
                    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
                {% endif %}
                {% if message['role'] == 'user' %}
                    {{ 'USER: ' + message['content'].strip() + '' }}
                {% elif message['role'] == 'assistant' %}
                    {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '' }}
                {% endif %}
                {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
                    {{ 'ASSISTANT: ' }}
                {% endfor %}"""
            cprint(f"[*]Set chat template to vicuna", "green")
        elif "llama-2" in model_path.lower():
            tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = false %}
{% endif %}
{% for message in loop_messages %}
{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
{% endif %}
{% if loop.index0 == 0 and system_message != false %}
{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
{% else %}
{% set content = message['content'] %}
{% endif %}
{% if message['role'] == 'user' %}
{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
{% elif message['role'] == 'assistant' %}
{{ ' ' + content.strip() + ' ' + eos_token }}
{% endif %}
{% if loop.last and message['role'] == 'user' and add_generation_prompt %}
{{ '[INST] ' }}
{% endif %}
{% endfor %}"""
            cprint(f"[*]Set chat template to llama2", "green")
        elif "llama-3.1" in model_path.lower():
            tokenizer.chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
            cprint(f"[*]Set chat template to llama3.1", "green")
        else:
            cprint("[-]The chat template for the tokenizer is not available.\n", "yellow")

    return tokenizer


def llm_adapter(model_name, mode="llm", **kwargs) -> Union[LocalLLM, LocalVLLM, LLM]:
    """
    Load LLM instance for model_name
    If model_name in MODEL_PATH, load local vLLM,
    otherwise load remote LLM API.
    """
    model_path = None
    if model_name in MODEL_PATH.keys():
        model_path = MODEL_PATH[model_name]
    elif os.path.exists(model_name):
        model_path = model_name

    if model_path is not None:
        if mode == "vllm":
            cprint(f"[*]Loading vLLM model ({model_name})...", "blue")
            return LocalVLLM(model_path, **kwargs)
        else:
            cprint(f"[*]Loading LLM model ({model_name})...", "blue")
            return LocalLLM(model_path, **kwargs)
    elif kwargs.get('device'):
        kwargs.pop("device")

    cprint(f"[*]Loading LLM API({model_name})...\n", "blue")
    # if re.match("^(gpt|claude|gemini|o1|grok|deepseek|glm|qwen)", model_name):
    if kwargs.get("base_url") is None:
        kwargs["base_url"] = HUIYAN_URL
    if kwargs.get("key") is None:
        if model_name.startswith(("o3-", "claude", "gpt")):
            kwargs["key"] = DIRECT_KEY
        # elif model_name.startswith(("glm")):
        #     kwargs["key"] = GUAN_KEY
        else:
            kwargs["key"] = DEFAULT_KEY
    return OpenAILLM(model_name, **kwargs)


def json_request(client: LLM, prompt: Union[str, list], repeat_num=3, verbose=True, **kwargs) -> dict:
    """请求LLM，返回Json格式数据"""
    assert client is not None
    # print(f"[*]Prompt:\n{prompt}") if verbose else None
    if isinstance(prompt, str):
        client.set_conv(LLM.user_prompt(prompt))
    elif isinstance(prompt, list):
        client.set_conv(prompt)
    else:
        cprint(f"[-]Json Request 提示词格式错误，应该为 [str, list]", "red")
        return None

    for i in range(repeat_num):
        final_res = {}
        results = client.generate(**kwargs)
        # print(f"[*]Raw output from {client.model_name}:\n{results}\n{'-'*99}")
        if results is None:
            return {}

        think_content = re.findall(r"<think>\n(.*)\n</think>", results)  # ds输出格式
        if len(think_content):
            final_res["think"] = think_content[0]

        json_contents = re.findall(r"```json([\s\S]*)```$", results, re.I)
        if len(json_contents):
            strip_content = json_contents[-1]
        else:
            strip_content = results[results.find("{"): results.rfind("}") + 1]

        try:
            json_res = json.loads(strip_content)
        except Exception as e:
            # print(f"[*] Failed to load JSON: {e}") if verbose else None
            json_res = None

        if not json_res:
            try:
                json_res = literal_eval(strip_content)
            except Exception as e:
                print(f"[-] Error in parsing response.\nRaw content:\n{'+' * 66}\n{results}\n{'+' * 66}\n") if verbose else None
                continue

        if not isinstance(json_res, dict):
            # cprint(f"[*] Failed to parse JSON: {type(json_res)}", "red") if verbose else None
            continue
        else:
            final_res = json_res

        if None not in list(final_res.values()):
            break

    return final_res


def json_request_batch(client: LLM, prompts: list, repeat_num=3, verbose=False, **kwargs) -> list[dict]:
    """请求一个批次的prompt 检查结果是否为Json 失败的再次请求"""
    assert client is not None
    assert prompts is not None

    client.clear_prompts()
    client.clear_conv()

    final_res = [{}] * len(prompts)
    todo_idx = list(range(len(prompts)))
    print(f"[*]Request {len(prompts)} prompts.") if verbose else None

    for i in range(repeat_num):
        if not todo_idx:
            break
        # cprint(f"Request the {i}th time, todo_idx:\n{todo_idx}\n", "green") if verbose else None

        for idx in todo_idx:
            if isinstance(prompts[idx], str):  # 字符串提示词
                conv = LLM.user_prompt(prompts[idx])
            elif isinstance(prompts[idx], list):  # 装配好的对话字典
                conv = prompts[idx]
            else:
                print(f"[*]对话内容格式错误，应该是str或者list[dict]")
                return None

            if isinstance(client, (LocalLLM, LocalVLLM)):
                client.add_conv_prompt(conv)
            else:
                client.add_conv(conv)

        results = client.generate_batch(**kwargs)

        succ_pos = []
        for pos, result in enumerate(results):
            json_contents = re.findall(r"```json([\s\S]*)```$", result, re.I)
            if len(json_contents):
                content = json_contents[-1]
            elif "{" in result:
                content = result[result.find("{"): result.rfind("}") + 1]
            else:
                content = result

            try:
                json_res = json.loads(content.replace("False", "false").replace("True", "true"))
            except Exception as e:
                # print(f"[*] Failed to load JSON: {e}") if verbose else None
                json_res = {}

            if not json_res:
                try:
                    json_res = literal_eval(content.replace("false", "False").replace("true", "True"))
                except Exception as e:
                    print(f"[-] Error in parsing response.\nRaw content:\n{'+'*99}\n{results}\n{'+'*99}\n") if verbose else None
                    continue

            if not isinstance(json_res, dict):
                cprint(f"[*] Failed to parse JSON: {result}", "yellow") if verbose else None
                # final_res[todo_idx[pos]] = {}
                continue
            else:
                final_res[todo_idx[pos]] = json_res
                succ_pos.append(todo_idx[pos])

        [todo_idx.remove(pos) for pos in succ_pos]
        if todo_idx:
            print(f"[*]部分请求不是json格式，todo_idx: {todo_idx}")

    return final_res


def request_batch(client: LLM, prompts=None, repeat_num=3, verbose=False, **kwargs) -> list:
    """请求一个批次的prompt 检查结果 失败的再次请求"""
    assert client is not None

    if isinstance(client, (LocalLLM, LocalVLLM)):
        client.clear_prompts()
    else:
        client.clear_conv()

    final_res = [""] * len(prompts)
    todo_idx = list(range(len(prompts)))
    for i in range(repeat_num):
        cprint(f"Request the {i}th time, todo_idx:\n{todo_idx}\n", "green") if verbose else None
        if not todo_idx:
            break

        for idx in todo_idx:
            if isinstance(prompts[idx], str):  # 字符串提示词
                conv = LLM.user_prompt(prompts[idx])
            elif isinstance(prompts[idx], list):  # 装配好的对话list
                conv = prompts[idx]
            else:
                print(f"[*]对话内容格式错误，应该是str或者list[dict]")
                return None

            if isinstance(client, (LocalLLM, LocalVLLM)):
                client.add_conv_prompt(conv)
            else:
                client.add_conv(conv)
        results = client.generate_batch(**kwargs)

        succ_pos = []
        for pos, result in enumerate(results):
            if (result or "").startswith("[!]Error:"):
                cprint(f"[*] Failed to request client: {result}", "red") if verbose else None
                continue
            else:
                final_res[todo_idx[pos]] = result
                succ_pos.append(todo_idx[pos])  # 删掉成功的位置

        [todo_idx.remove(pos) for pos in succ_pos]

    return final_res



if __name__ == "__main__":
    model = "glm-4"
    # glm4 = GLMLLM(model, "d4aef14b47f6e16f17ce62b21f85dc74.oL4MbqPu7jLKXGNU", repeat_num=1)
    # glm4.set_conv(LLM.user_prompt("1+1=?"))
    # response = glm4.generate()
    # print(response)
