import ws
import asyncio
import ujson
import ubinascii
from collections import namedtuple
import time

# Define a namedtuple for chunks received from the OpenAI Realtime API
# This will hold either text or audio data, and a flag for end of turn.
Chunk = namedtuple("Chunk", ["text", "audio", "is_end_turn"])

class OpenAISession:
    """
    Manages a single OpenAI Realtime API session over a WebSocket connection.
    This class handles the high-level interaction for sending text prompts
    and voice chunks, and receiving voice responses and text transcripts
    according to the OpenAI Realtime API WebSockets specification.
    """
    def __init__(self, websocket_client: ws.AsyncWebsocketClient, vad_disabled: bool = False):
        """
        Initializes an OpenAISession.
        :param websocket_client: An instance of AsyncWebsocketClient connected to the OpenAI Realtime API endpoint.
        :param vad_disabled: If True, Voice Activity Detection will be disabled for this session,
                             requiring manual input commitment and response triggering.
        """
        self.ws = websocket_client
        self.session_active = False # Flag to indicate if the session is currently active
        self.vad_disabled = vad_disabled # Store VAD state

    async def _send_json(self, payload: dict) -> bool:
        """
        Helper method to serialize a dictionary to JSON and send it as a WebSocket text frame.
        :param payload: The dictionary to send.
        :return: True if sent successfully, False otherwise.
        """
        if not await self.ws.open():
            print("OpenAISession: Cannot send JSON, WebSocket not open.")
            return False
        try:
            json_str = ujson.dumps(payload)
            await self.ws.send(json_str) # OpenAI Realtime API expects JSON messages as text frames
            return True
        except Exception as e:
            print(f"OpenAISession: Error sending JSON payload: {e}")
            print(self.ws.last_payload)
            return False

    async def _receive_json(self) -> dict | None:
        """
        Helper method to receive a WebSocket text frame and parse it as JSON.
        It continuously tries to receive until valid JSON is found or connection closes.
        :return: The parsed JSON dictionary, or None if connection closes or an unrecoverable error occurs.
        """
        while True:
            raw_data = await self.ws.recv()
            if isinstance(raw_data, bytes):
              raw_data = raw_data.decode()
            if raw_data is None:
                print("OpenAISession: Connection Closed.")
                return None # Connection closed or error during receive
            if isinstance(raw_data, str): # Expecting text frames for JSON messages
                try:
                    return ujson.loads(raw_data)
                except ValueError as e:
                    print(f"OpenAISession: Error parsing JSON: {e}, Data: '{raw_data}'")
                    # Continue waiting for valid JSON if parsing fails
                    continue
            else:
                print(f"OpenAISession: Received non-text data (type: {type(raw_data)}). Ignoring, expecting JSON.")
                # Continue waiting for text/JSON
                continue

    async def add_text_chunk(self, txt: str, turnComplete: bool = False) -> bool:
        """
        Sends a text input as a conversation item to the OpenAI Realtime API.
        If VAD is disabled and turnComplete is True, it will automatically trigger a response.
        :param txt: The text string to send.
        :param turnComplete: If True, signals the end of the user's turn and triggers a response
                             (only applicable when VAD is disabled).
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("OpenAISession not active. Call OpenAIClient.start_session() first.")
            return False
        print(f"OpenAISession: Sending text input: '{txt}' (turnComplete={turnComplete}).")

        payload = {
          "type": "conversation.item.create",
          "item": {
            "type": "message",
            "role": "user",
            "content": [
              {
                "type": "input_text",
                "text": txt,
              }
            ]
          },
        }
        success = await self._send_json(payload)

        if success and self.vad_disabled and turnComplete:
            print("OpenAISession: VAD disabled and turnComplete=True. Triggering response.")
            await self._trigger_response(modalities=["text", "audio"]) # Default to both text and audio
        return success

    async def add_voice_chunk(self, voice_bytes: bytes, turnComplete: bool = False) -> bool:
        """
        Sends a voice chunk (raw bytes) to the OpenAI Realtime API's audio buffer.
        If VAD is disabled and turnComplete is True, it will automatically commit the audio,
        trigger a response, and clear the audio buffer for the next turn.
        :param voice_bytes: The raw audio bytes to send.
        :param turnComplete: If True, signals the end of the user's turn by committing audio,
                             triggering a response, and clearing the buffer (only applicable when VAD is disabled).
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("OpenAISession not active. Call OpenAIClient.start_session() first.")
            return False
        print(f"OpenAISession: Sending audio chunk of {len(voice_bytes)} bytes to buffer (turnComplete={turnComplete}).")
        # Base64 encode the voice bytes. .decode('utf-8') converts bytes to string, .strip() removes potential newline.
        encoded_voice_data = ubinascii.b2a_base64(voice_bytes).decode('utf-8').strip()
        
        payload = {
            "type": "input_audio_buffer.append",
            "audio": encoded_voice_data
        }
        success = await self._send_json(payload)

        if success and self.vad_disabled and turnComplete:
            print("OpenAISession: VAD disabled and turnComplete=True. Committing audio, triggering response, and clearing buffer.")
            await self._commit_audio_input()
            await self._trigger_response(modalities=["text", "audio"]) # Default to both text and audio
            await self._clear_audio_buffer() # Prepare for the next turn
        return success

    async def _commit_audio_input(self) -> bool:
        """
        Commits the currently buffered audio input to the OpenAI Realtime API.
        This signals the end of a user's audio input segment when VAD is disabled.
        (Internal method, use send_audio_chunk with turnComplete=True for automation).
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("OpenAISession not active. Call OpenAIClient.start_session() first.")
            return False
        print("OpenAISession: Committing audio input.")
        payload = {
            "type": "input_audio_buffer.commit"
        }
        return await self._send_json(payload)

    async def _clear_audio_buffer(self) -> bool:
        """
        Clears the audio input buffer. Useful when VAD is disabled, before starting
        a new audio input sequence.
        (Internal method, use send_audio_chunk with turnComplete=True for automation).
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("OpenAISession not active. Call OpenAIClient.start_session() first.")
            return False
        print("OpenAISession: Clearing audio buffer.")
        payload = {
            "type": "input_audio_buffer.clear"
        }
        return await self._send_json(payload)

    async def _trigger_response(self, modalities: list = None) -> bool:
        """
        Triggers a model response from the OpenAI Realtime API.
        This is typically called after sending text input or committing audio input
        when VAD is disabled.
        (Internal method, use send_text_input or send_audio_chunk with turnComplete=True for automation).
        :param modalities: Optional list of modalities for the response (e.g., ["text"], ["audio"], ["text", "audio"]).
                           Defaults to both text and audio if not specified.
        :return: True if sent successfully, False otherwise.
        """
        print("trigger response")
        if not self.session_active:
            print("OpenAISession not active. Call OpenAIClient.start_session() first.")
            return False
        print("OpenAISession: Triggering model response.")
        payload = {
            "type": "response.create",
            "response": {}
        }
        if modalities:
            payload["response"]["modalities"] = modalities
        return await self._send_json(payload)

    async def next_chunk(self) -> Chunk:
        """
        Waits for and returns the next chunk (audio or text) from the OpenAI Realtime API.
        This method continuously receives messages, parses them, and specifically
        looks for 'speech' (audio) and 'chunk' (text) events.
        :return: A Chunk namedtuple containing the received text or audio,
                 or an empty Chunk if the connection closes or an error occurs.
        """
        empty_chunk = Chunk(text=None, audio=None, is_end_turn=True)
        if not self.session_active:
            print("OpenAISession: Not active. Call OpenAIClient.start_session() first.")
            return empty_chunk

        print("OpenAISession: Waiting for OpenAI Realtime API response...")
        while True:
            server_message = await self._receive_json() # Use helper to get parsed JSON
            if server_message is None:
                print("OpenAISession: WebSocket connection closed or error during receive. Ending session.")
                self.session_active = False
                return empty_chunk

            message_type = server_message.get("type")

            if message_type == "response.audio.delta":
                # This message type contains audio data
                if "delta" in server_message:
                    try:
                        voice_bytes = ubinascii.a2b_base64(server_message["delta"])
                        print(f"OpenAISession: Received speech chunk of {len(voice_bytes)} bytes.")
                        # is_end_turn is False for speech chunks as they are part of a continuous stream
                        return Chunk(text=None, audio=voice_bytes, is_end_turn=False)
                    except Exception as e:
                        print(f"OpenAISession: Error decoding base64 audio data: {e}")
                        continue # Continue waiting for valid data
                else:
                    print(f"OpenAISession: 'speech' message without 'audio' data: {server_message}. Ignoring.")
                    continue
            elif message_type == "chunk":
                # This message type contains text data (transcription or model response)
                if "text" in server_message:
                    text_data = server_message["text"]
                    print(f"OpenAISession: Received text chunk: '{text_data}'.")
                    # is_end_turn is False for text chunks as they are part of a continuous stream
                    return Chunk(text=text_data, audio=None, is_end_turn=False)
                else:
                    print(f"OpenAISession: 'chunk' message without 'text' data: {server_message}. Ignoring.")
                    continue
            elif message_type == "response.output_item.done":
                # Explicit signal for end of a turn
                print("OpenAISession: Received 'turn.end' message. Signaling end of turn.")
                return Chunk(text=None, audio=None, is_end_turn=True)
            elif message_type == "ping":
                # Ping message, typically for keeping connection alive
                print("OpenAISession: Received 'ping' message. Ignoring.")
                continue
            elif message_type == "pong":
                # Pong message, response to ping
                print("OpenAISession: Received 'pong' message. Ignoring.")
                continue
            elif message_type == "error":
                # Error message from the API
                print(f"OpenAISession: Received error message: {server_message}. Ending session.")
                self.session_active = False
                await self.ws.close()
                return empty_chunk
            elif message_type == "session.updated":
                print(f"OpenAISession: Received 'session.updated' message: {server_message}. Session configuration updated.")
                continue
            elif message_type == "response.done":
                # When VAD is disabled, this might be the final signal for a response.
                print(f"OpenAISession: Received 'response.done' message: {server_message}.")
                # You might want to extract final text from here if needed, but next_chunk focuses on deltas.
                # For now, we'll just log it and continue to look for other chunks.
                continue
            else:
                print(f"OpenAISession: Received unhandled server message type: '{message_type}'. Full message: {server_message}. Waiting for next message.")
                continue
            

    async def end_session(self) -> None:
        """
        Ends the current OpenAI Realtime API session and closes the underlying WebSocket connection.
        """
        if self.session_active:
            print("OpenAISession: Ending session and closing WebSocket.")
            await self.ws.close()
            self.session_active = False
        else:
            print("OpenAISession: No active session to end.")


class OpenAIClient:
    """
    Client for interacting with the OpenAI Realtime API.
    Manages the creation of new OpenAISession instances and the underlying WebSocket connection.
    """
    # Official WebSocket URI for OpenAI Realtime API
    DEFAULT_OPENAI_REALTIME_URI = "wss://api.openai.com/v1/realtime"

    def __init__(self, api_key: str, model: str, websocket_uri: str = DEFAULT_OPENAI_REALTIME_URI, 
                 vad_disabled: bool = True, **websocket_kwargs):
        """
        Initializes the OpenAIClient.
        :param api_key: Required. Your OpenAI API key.
        :param model: Required. The Realtime model ID to connect to (e.g., "gpt-4o-realtime-preview-2024-12-17").
        :param websocket_uri: The WebSocket URI for the OpenAI Realtime API endpoint.
                              Defaults to the official OpenAI Realtime API endpoint.
        :param vad_disabled: If True, Voice Activity Detection will be disabled for this session,
                             requiring manual input commitment and response triggering.
        :param websocket_kwargs: Additional keyword arguments to pass to AsyncWebsocketClient
                                 (e.g., certfile, keyfile, cafile, cert_reqs).
        """
        self.api_key = api_key
        self.websocket_uri = websocket_uri
        self.websocket_kwargs = websocket_kwargs
        self.model = model
        self.vad_disabled = vad_disabled # Store VAD state
        self._current_ws_client = None # Stores the active websocket client for a session

    async def start_session(self, system_instruction: str | None = None) -> OpenAISession | None:
        """
        Starts a new OpenAI Realtime API session.
        Establishes a WebSocket connection and configures VAD if specified.
        :return: An OpenAISession instance if connection and setup are successful, None otherwise.
        """
        print(f"OpenAIClient: Starting new session to {self.websocket_uri} with model '{self.model}'...")
        self._current_ws_client = ws.AsyncWebsocketClient(**self.websocket_kwargs) # Pass kwargs to WS client
        session = OpenAISession(self._current_ws_client, vad_disabled=self.vad_disabled) # Pass VAD state to session

        try:
            # Prepare WebSocket URI with model query parameter
            uri = f"{self.websocket_uri}?model={self.model}"
            
            # Prepare headers for authentication and beta flag
            headers = [
                ("Authorization", f"Bearer {self.api_key}"),
                ("OpenAI-Beta", "realtime=v1"),
            ]

            print(f"OpenAIClient: Connecting to {uri} with headers...")
            await self._current_ws_client.handshake(uri, headers=headers) # Pass headers during handshake
            
            if not await self._current_ws_client.open():
                print("OpenAIClient: WebSocket handshake failed, connection not open.")
                await session.end_session() # Clean up
                return None

            print("OpenAIClient: WebSocket connection established. Session is active.")
            session.session_active = True # Mark the session as active

            if self.vad_disabled:
                print("OpenAIClient: VAD disabled. Sending session.update to configure turn_detection to null.")
                update_payload = {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None, # Disable VAD
                        "instructions": system_instruction,
                        "input_audio_transcription": {
                            "model": "whisper-1"
                        },
                    },
                }
                if not await session._send_json(update_payload):
                    print("OpenAIClient: Failed to send VAD disable update.")
                    await session.end_session()
                    return None
                
                # Wait for session.updated confirmation
                print("OpenAIClient: Waiting for session.updated confirmation...")
                while True:
                    server_message = await session._receive_json()
                    if server_message is None:
                        print("OpenAIClient: Connection closed before session.updated confirmation.")
                        await session.end_session()
                        return None
                    if server_message.get("type") == "session.updated":
                        print(f"OpenAIClient: session.updated received. VAD successfully disabled. {server_message}")
                        break
                    else:
                        print(f"OpenAIClient: Received unexpected message during VAD setup: {server_message}. Waiting for session.updated.")
                        # Continue waiting for session.updated or an error

            return session

        except Exception as e:
            print(f"OpenAIClient: Failed to start session due to an exception: {e}")
            await session.end_session() # Ensure cleanup on failure
            raise
            return None

    async def close_client(self):
        """
        Closes the underlying WebSocket client if it's open.
        Useful for cleaning up resources when the OpenAIClient itself is no longer needed.
        """
        if self._current_ws_client and await self._current_ws_client.open():
            print("OpenAIClient: Closing underlying WebSocket client.")
            await self._current_ws_client.close()
            self._current_ws_client = None
        else:
            print("OpenAIClient: No active WebSocket client to close.")
