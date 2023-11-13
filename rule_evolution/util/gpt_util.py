from typing import List, Dict, Any
from aiohttp import ClientSession, ClientTimeout
from dataclasses import dataclass
from threading import Thread
import asyncio
from asyncio import Lock, Condition, Future
import logging
import time
import heapq
from enum import Enum, auto
#from llama_cpp import Llama

from openai import OpenAI
import util

# --- LOGGERS ---
logger = util.get_basic_logger('gpt_util.py')
logger.setLevel(logging.ERROR)

# --- MSG FORMATTERS ---
def format_gpt_msg(role: str, content: str) -> Dict[str, str]:
    """
    Formats a string into a GPT message with the specified role.

    Parameters:
        role: The role associated with the message ('system', 'user', or 'assistant').
        content: The string content of the message.
    
    Returns:
        A GPT message with the specified parameters
        i.e. a dictionary mapping the role and content.
    """
    assert role in ['system', 'user', 'assistant']
    assert isinstance(content, str)

    return {'role' : role, 'content': content}


def format_gpt_msgs(roles: List[str], contents: List[str]) -> List[Dict[str, str]]:
    """
    Formats a list of strings into GPT messages with the specified roles.
    Roles and contents must match in length.

    Parameters:
        roles: The roles associated with each message ('system', 'user', or 'assistant').
        contents: The string content for each message.
    
    Returns:
        A list of GPT messages i.e. a list of dictionaries.
    """
    assert len(roles) == len(contents)

    return [format_gpt_msg(roles[i], contents[i]) for i in range(len(roles))]


def format_gpt_msgs_single_role(role: str, contents: List[str]) -> List[Dict[str, str]]:
    """
    Formats a list of strings into GPT messages with the specified role.

    Parameters:
        role: The role associated with all messages ('system', 'user', or 'assistant').
        contents: The string content for each message.

    Returns:
        A list of GPT messages i.e. a list of dictionaries.
    """
    return format_gpt_msgs([role] * len(contents), contents)


# --- GPT CLIENT ---
@dataclass
class GPTEndpointRequest:
    # Attributes
    endpoint: str
    json: Dict[str, Any]
    headers: Dict[str, Any]
    last_timeout_time: float
    last_fail_delay_time: float

    future: Future = None
    last_request_time: float = None
    last_fail_time: float = None

    def __lt__(self, other):
        self_time = self.last_request_time
        if self.last_fail_time != None:
            self_time = self.last_fail_time + self.last_fail_delay_time
        
        other_time = other.last_request_time
        if other.last_fail_time != None:
            other_time = other.last_fail_time + other.last_fail_delay_time
            
        return self_time > other_time # Failed requests have higher priority


class LLMType(Enum):
    OPENAI = auto()
    LLAMA = auto()


class LLMClient:
    # --- ENDPOINTS ---
    CHAT_COMPLETION_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self,
                 llm_type: LLMType = LLMType.OPENAI,
                 api_secret_key: str = None,
                 llama_path: str = None):
        """
        Constructs a LLMClient object.

        Parameters:
            llm_type: Type of LLM (GPT or Llama).
            api_secret_key: Key for the ChatGPT API.
            initial_timeout_time_s: Number of seconds to wait before request timeout. Recommended: 40s for gpt-3.5-turbo.
            initial_fail_delay_time_s: Number of seconds to wait before serving request again. Recommended: 20s for gpt-3.5-turbo.
            llama_path: Path to local llama.
        """
        assert llm_type in LLMType
        self.llm_type = llm_type

        if llm_type == LLMType.OPENAI:
            assert api_secret_key != None

            self.api_key = api_secret_key
            self.internal_client = OpenAI(api_key=api_secret_key)

            # Start serving requests
            self.async_loop = asyncio.new_event_loop()
            self.async_loop.run_until_complete(self._setup_queue())
            self.serve_thread = Thread(target=lambda: self.async_loop.run_until_complete(self._serve_queue()))
            self.serve_thread.start()
        elif llm_type == LLMType.LLAMA:
            assert llama_path != None
            self.internal_client = Llama(model_path=llama_path, chat_format='llama-2', n_ctx=2048)


    def __enter__(self):
        """
        Enable 'with' use.
        """
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Enable 'with' use.
        """
        if self.llm_type == LLMType.OPENAI:
            self.close()


    def close(self):
        """
        Ends client and joins child threads belonging to the client.
        Must be called when finished with client.
        """
        asyncio.run_coroutine_threadsafe(self._end_queue(), self.async_loop)
        self.serve_thread.join()


    async def _end_queue(self):
        """
        Waits for all remaining requests to be served and then notifies
        queue server to stop waiting for requests.
        """
        async with self.request_queue_lock:
            await self.request_queue_cv.wait_for(lambda: len(self.request_queue) == 0)
            self.stop = True
            self.request_queue_cv.notify()

    
    @property
    def api_header(self):
        return {'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self.api_key}
    

    async def _add_request(self, request: GPTEndpointRequest, prepend: bool = False):
        """
        Adds a request to the request queue.

        Parameters:
            request: A GPTEndpointRequest
            prepend: Add request to front of queue if True, append otherwise
        """
        async with self.request_queue_lock:
            request.last_request_time = time.time()
            heapq.heappush(self.request_queue, request)
            self.request_queue_cv.notify()
    

    async def _serve_request(self, request: GPTEndpointRequest):
        """
        Serves given request by sending request to the specified endpoint.
        Request's future is populated upon endpoint response.
        Upon request error, request is sent to front of the queue with new fail time.

        Parameters:
            request: A GPTEndpointRequest
        """
        try:
            timeout = ClientTimeout(total=request.last_timeout_time)
            async with ClientSession(timeout=timeout) as session:
                # Send request
                response = await session.post(request.endpoint, 
                                            json=request.json, 
                                            headers=request.headers)
                response = await response.json()

                if 'error' in response:
                    raise Exception(str(response))
                
                # Fulfill response
                request.future.set_result(response)
        except BaseException as e:
            logger.info(f'Failed to serve request: {e}. Request: {request}')
            
            # Add request back in queue
            if request.last_fail_time == None: # Only backoff after first fail
                request.last_timeout_time *= 2
                request.last_fail_delay_time *= 2
            request.last_fail_time = time.time()
            await self._add_request(request, prepend=True)
                

    async def _serve_queue(self):
        """
        Continually serves the request queue until signaled to stop using close().
        """
        while True:
            # Wait for request
            async with self.request_queue_lock:
                if len(self.request_queue) == 0:
                    self.request_queue_cv.notify() # Notify destructor wait
                    await self.request_queue_cv.wait()
                    if self.stop:
                        break

                # Get first request
                request = heapq.heappop(self.request_queue)

            # Wait for fail delay 
            if request.last_fail_time != None:
                time_elapsed = time.time() - request.last_fail_time
                wait_s = request.last_fail_delay_time - time_elapsed
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

            # Serve request (non-blocking)
            asyncio.create_task(self._serve_request(request))


    async def _setup_queue(self):
        """
        Separate function to init asyncio queue properties.
        Required so that all queue serving belongs to one event loop.
        """
        self.request_queue: List[GPTEndpointRequest] = []
        self.request_queue_lock = Lock()
        self.request_queue_cv = Condition(self.request_queue_lock)
        self.stop = False


    async def _bridge_future(self, future) -> Any:
        """
        Separate function to await asyncio future.
        Required so that future awaiting and fulfillment belong to one event loop.
        """
        return await future


    async def _get_request_result(self, request: GPTEndpointRequest) -> Any:
        """
        Awaits and returns response to given request.
        
        Parameters:
            request: A GPTEndpointRequest with no future attached
        
        Returns:
            The request's response.
        """
        request.future = self.async_loop.create_future()
        asyncio.run_coroutine_threadsafe(self._add_request(request), self.async_loop)
        response = asyncio.run_coroutine_threadsafe(self._bridge_future(request.future), self.async_loop)
        return await asyncio.wrap_future(response)


    async def get_completion(self, messages: List[Dict[str, str]], model: str = None) -> str:
        """
        Pings the GPT API to complete a chat with history 'messages'.
        
        Parameters:
            messages: List of GPT messages i.e. list of dictionaries.
            model: String identifier of the GPT model to ping.

        Returns:
            The chat completion.
        """
        
        # Chat completion successful response
        if self.llm_type == LLMType.OPENAI:
            assert model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview']
            
            # Initial timeouts and delays
            if model == 'gpt-3.5-turbo':
                initial_timeout_time_s = 60
                initial_fail_delay_time_s = 30
            else:
                initial_timeout_time_s = 120
                initial_fail_delay_time_s = 60

            # Chat completion endpoint request
            request = GPTEndpointRequest(self.CHAT_COMPLETION_ENDPOINT,
                                        {"model": model, "messages": messages},
                                        self.api_header,
                                        initial_timeout_time_s,
                                        initial_fail_delay_time_s)
        
            response = await self._get_request_result(request)
        elif self.llm_type == LLMType.LLAMA:
            response = self.internal_client.create_chat_completion(messages)

        return response['choices'][0]['message']['content']



    async def get_one_shot_completion(self, role: str, intro: str, example: str, model: str = None) -> str:
        """
        Pings GPT chat completion on a one-shot context (intro) and message (example).

        Parameters:
            role: The role associated with the intro and example ('system', 'user', or 'assistant').
            intro: The context for the 'example'.
            example: The prompt for chat completion.
            model: String identifier of the GPT model to ping.

        Returns:
            The chat completion.
        """
        return await self.get_completion(format_gpt_msgs_single_role(role, [intro, example]), model)
    

"""
client = LLMClient(llm_type=LLMType.LLAMA, llama_path='../../models/llama-2-13b-chat.Q8_0.gguf')
asyncio.run(client.get_one_shot_completion('user', 'hello', 'output a tic tac toe board'))
"""