import ws
import asyncio
import ujson
import ubinascii
from collections import namedtuple
import time

Chunk = namedtuple("Chunk", ["text", "audio", "bitrate", "is_end_turn"])

class GeminiSession:
    """
    Manages a single AI Studio Live session over a WebSocket connection.
    This class handles the high-level interaction for sending text prompts
    and voice chunks, and receiving voice responses according to the
    AI Studio Live WebSockets API specification.
    """
    def __init__(self, websocket_client: ws.AsyncWebsocketClient):
        """
        Initializes a GeminiSession.
        :param websocket_client: An instance of AsyncWebsocketClient connected to the AI Studio Live endpoint.
        """
        self.ws = websocket_client
        self.session_active = False # Flag to indicate if the session is currently active

    async def _send_json(self, payload: dict) -> bool:
        """
        Helper method to serialize a dictionary to JSON and send it as a WebSocket text frame.
        :param payload: The dictionary to send.
        :return: True if sent successfully, False otherwise.
        """
        if not await self.ws.open():
            print("GeminiSession: Cannot send JSON, WebSocket not open.")
            return False
        try:
            json_str = ujson.dumps(payload)
            await self.ws.send(json_str) # AI Studio Live API expects JSON messages as text frames
            return True
        except Exception as e:
            print(f"GeminiSession: Error sending JSON payload: {e}")
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
            #print(f"incoming {raw_data}")
            if isinstance(raw_data, bytes):
              raw_data = raw_data.decode()
            if raw_data is None:
                print("Connection Closed: ", self.ws.last_payload)
                return None # Connection closed or error during receive
            if isinstance(raw_data, str): # Expecting text frames for JSON messages
                try:
                    return ujson.loads(raw_data)
                except ValueError as e:
                    print(f"GeminiSession: Error parsing JSON: {e}, Data: '{raw_data}'")
                    # Continue waiting for valid JSON if parsing fails
                    continue
            else:
                print(f"GeminiSession: Received non-text data (type: {type(raw_data)}). Ignoring, expecting JSON.")
                # Continue waiting for text/JSON
                continue

    async def add_text_chunk(self, txt: str, turnComplete: bool = False) -> bool:
        """
        Sends a text prompt to the AI Studio Live API as a BidiGenerateContentRealtimeInput message.
        :param txt: The text string to send.
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("GeminiSession not active. Call GeminiClient.start_session() first.")
            return False
        print(f"GeminiSession: Sending text chunk: '{txt}'")

        payload = {
          'clientContent': {
            "turns": [{
              "parts": [
                { "text": txt },
              ],
              "role": "user",
            }],
            "turnComplete": turnComplete,
          }
        }
        return await self._send_json(payload)

    async def add_voice_chunk(self, voice_bytes: bytes, mime_type: str = "audio/wav", turnComplete: bool = False) -> bool:
        """
        Sends a voice chunk (raw bytes) to the AI Studio Live API as a BidiGenerateContentRealtimeInput message.
        The voice data is base64 encoded for transmission within the JSON payload.
        :param voice_bytes: The raw audio bytes to send.
        :param mime_type: The MIME type of the audio data (e.g., "audio/wav", "audio/ogg").
                          Defaults to "audio/wav".
        :return: True if sent successfully, False otherwise.
        """
        if not self.session_active:
            print("GeminiSession not active. Call GeminiClient.start_session() first.")
            return False
        print(f"GeminiSession: Sending voice chunk of {len(voice_bytes)} bytes (MIME: {mime_type}).")
        # Base64 encode the voice bytes. .decode('utf-8') converts bytes to string, .strip() removes potential newline.
        encoded_voice_data = ubinascii.b2a_base64(voice_bytes).decode('utf-8').strip()
        # payload = {
        #     "realtimeInput": {
        #         "audio": {
        #             "data": encoded_voice_data,
        #             "mimeType": mime_type
        #         }
        #     }
        # }
        payload = {
          'clientContent': {
            "turns": [{
              "parts": [{
                "inlineData": {
                  "data": encoded_voice_data,
                  "mimeType": mime_type,
                },
              }],
              "role": "user",
            }],
            "turnComplete": turnComplete,
          }
        }
        return await self._send_json(payload)

    async def next_chunk(self) -> Chunk:
        """
        Waits for and returns the next voice chunk from the AI Studio Live API.
        This method continuously receives messages, parses them, and specifically
        looks for base64-encoded audio blobs within 'serverContent.modelTurn.parts'.
        It will ignore other message types (like text parts, metadata, tool calls)
        and continue waiting until voice data is found or the connection closes.
        :return: The received voice bytes, or an empty bytes object if the connection closes,
                 an error occurs, or no voice data is found in the response.
        """
        empty_chunk = Chunk(
          text=None, audio=None, bitrate=None, is_end_turn=True
        )
        if not self.session_active:
            print("GeminiSession: Not active. Call GeminiClient.start_session() first.")
            return empty_chunk

        print("GeminiSession: Waiting for AI Studio Live response (voice bytes in serverContent)...")
        while True:
            server_message = await self._receive_json() # Use helper to get parsed JSON
            if server_message is None:
                print("GeminiSession: WebSocket connection closed or error during receive. Ending session.")
                self.session_active = False
                return empty_chunk

            if "serverContent" in server_message:
                server_content = server_message["serverContent"]
                chunk_audio = b""
                chunk_is_end_turn=False
                if "turnComplete" in server_content:
                  chunk_is_end_turn = server_content['turnComplete']
                  # Special handling for "turnComplete" messages that might not have audio
                  if chunk_is_end_turn and not server_content.get("modelTurn"):
                    print("GeminiSession: Received turnComplete without modelTurn, indicating end of turn.")
                    return Chunk(text=None, audio=b"", bitrate=None, is_end_turn=True)

                if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                    for part in server_content["modelTurn"]["parts"]:
                        print("part", list(part.keys()))
                        if "inlineData" in part and "data" in part["inlineData"] and "mimeType" in part["inlineData"]:
                            try:
                                # Base64 decode the voice bytes
                                voice_bytes = ubinascii.a2b_base64(part["inlineData"]["data"])
                                print(f"GeminiSession: Received voice chunk of {len(voice_bytes)} bytes (MIME: {part['inlineData']['mimeType']}).")
                                chunk_audio += voice_bytes
                            except Exception as e:
                                print(f"GeminiSession: Error decoding base64 voice data: {e}")
                                # Continue to next part or next message if decoding fails
                                continue
                        elif "text" in part:
                            print(f"GeminiSession: Received text part in modelTurn: '{part['text']}'. Ignoring, expecting voice bytes.")
                            continue
                else:
                    print(f"GeminiSession: serverContent does not contain expected modelTurn/parts. Message: {server_content}. Waiting for next message.")
                    continue
                return Chunk(text=None, audio=chunk_audio, bitrate=24000, is_end_turn=chunk_is_end_turn)
            elif "setupComplete" in server_message:
                print("GeminiSession: Received setupComplete (unexpected during response wait). Ignoring.")
                continue
            elif "toolCall" in server_message:
                print(f"GeminiSession: Received toolCall: {server_message['toolCall']}. Tool calls are not handled in this simple client. Ignoring.")
                # In a full client, you'd process tool calls and send ToolResponse
                continue
            elif "goAway" in server_message:
                print(f"GeminiSession: Received GoAway message: {server_message['goAway'].get('timeLeft', 'N/A')}. Server is disconnecting. Ending session.")
                self.session_active = False
                await self.ws.close()
                return empty_chunk
            elif "usageMetadata" in server_message:
                print(f"GeminiSession: Received usageMetadata: {server_message['usageMetadata']}. Ignoring.")
                continue
            else:
                print(f"GeminiSession: Received unhandled server message type: {list(server_message.keys())}. Full message: {server_message}. Waiting for next message.")
                continue
            

    async def end_session(self) -> None:
        """
        Ends the current Gemini session and closes the underlying WebSocket connection.
        """
        if self.session_active:
            print("GeminiSession: Ending session and closing WebSocket.")
            await self.ws.close()
            self.session_active = False
        else:
            print("GeminiSession: No active session to end.")


class GeminiClient:
    """
    Client for interacting with the AI Studio Live API.
    Manages the creation of new GeminiSession instances and the underlying WebSocket connection.
    """
    # Official WebSocket URI for AI Studio Live
    DEFAULT_GEMINI_LIVE_URI = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    def __init__(self, model: str, websocket_uri: str = DEFAULT_GEMINI_LIVE_URI, 
                 generation_config: dict = None, tools: list = None,
                 realtime_input_config: dict = None,
                 input_audio_transcription: dict = None,
                 output_audio_transcription: dict = None,
                 **websocket_kwargs):
        """
        Initializes the GeminiClient.
        :param model: Required. The model's resource name (e.g., "models/gemini-pro").
        :param websocket_uri: The WebSocket URI for the AI Studio Live endpoint.
                              Defaults to the official Gemini Live API endpoint.
        :param generation_config: Optional. Configuration for content generation.
        :param tools: Optional. List of tools the model may use.
        :param realtime_input_config: Optional. Configures handling of real-time input.
        :param input_audio_transcription: Optional. If set, enables transcription of voice input.
        :param output_audio_transcription: Optional. If set, enables transcription of model's audio output.
        :param websocket_kwargs: Additional keyword arguments to pass to AsyncWebsocketClient
                                 (e.g., certfile, keyfile, cafile, cert_reqs).
        """
        self.websocket_uri = websocket_uri
        self.websocket_kwargs = websocket_kwargs
        self.model = model
        self.generation_config = generation_config
        self.tools = tools
        self.realtime_input_config = realtime_input_config
        self.input_audio_transcription = input_audio_transcription
        self.output_audio_transcription = output_audio_transcription
        self._current_ws_client = None # Stores the active websocket client for a session

    async def start_session(self, api_key: str, system_instruction: str = None) -> GeminiSession | None:
        """
        Starts a new Gemini AI Studio Live session.
        Establishes a WebSocket connection and sends the initial session configuration.
        :param api_key: Required. Your API key for authentication.
        :param system_instruction: Optional. System instructions for the model.
        :return: A GeminiSession instance if connection and setup are successful, None otherwise.
        """
        print(f"GeminiClient: Starting new session to {self.websocket_uri} with model '{self.model}'...")
        self._current_ws_client = ws.AsyncWebsocketClient(**self.websocket_kwargs) # Pass kwargs to WS client
        session = GeminiSession(self._current_ws_client) # Create session early to use its helpers

        try:
            # Perform WebSocket handshake
            uri = self.websocket_uri + "?key="+api_key
            print(f"connecting to {uri}")
            await self._current_ws_client.handshake(uri) # Handshake takes URI, kwargs handled in client init
            if not await self._current_ws_client.open():
                print("GeminiClient: WebSocket handshake failed, connection not open.")
                await session.end_session() # Clean up
                return None

            # Prepare initial BidiGenerateContentSetup message
            setup_payload = {
                "setup": {
                    "model": self.model,
                }
            }

            if self.generation_config:
                setup_payload["setup"]["generationConfig"] = self.generation_config
            if system_instruction: # system_instruction is still passed to start_session
                # System instruction is a Content object, which has a 'parts' field
                setup_payload["setup"]["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            if self.tools:
                setup_payload["setup"]["tools"] = self.tools
            if self.realtime_input_config:
                setup_payload["setup"]["realtimeInputConfig"] = self.realtime_input_config
            if self.input_audio_transcription:
                setup_payload["setup"]["inputAudioTranscription"] = self.input_audio_transcription
            if self.output_audio_transcription:
                setup_payload["setup"]["outputAudioTranscription"] = self.output_audio_transcription

            print("GeminiClient: Sending session setup message...")
            if not await session._send_json(setup_payload):
                print("GeminiClient: Failed to send setup message.")
                await session.end_session()
                return None

            # Wait for BidiGenerateContentSetupComplete message from the server
            print("GeminiClient: Waiting for setup completion message...")
            while True:
                server_message = await session._receive_json()
                if server_message is None:
                    print("GeminiClient: Connection closed before setup complete.")
                    await session.end_session()
                    return None
                if "setupComplete" in server_message:
                    print("GeminiClient: Setup complete received. Session is active.")
                    session.session_active = True # Mark the session as active
                    return session
                else:
                    # Log any unexpected messages received during the setup phase
                    print(f"GeminiClient: Received unexpected message during setup: {server_message}. Waiting for setupComplete.")
                    # In a more robust client, you might handle 'goAway' or other
                    # early messages here, but for this simple case, we just wait.

        except Exception as e:
            print(f"GeminiClient: Failed to start session due to an exception: {e}")
            await session.end_session() # Ensure cleanup on failure
            return None

    async def close_client(self):
        """
        Closes the underlying WebSocket client if it's open.
        Useful for cleaning up resources when the GeminiClient itself is no longer needed.
        """
        if self._current_ws_client and await self._current_ws_client.open():
            print("GeminiClient: Closing underlying WebSocket client.")
            await self._current_ws_client.close()
            self._current_ws_client = None
        else:
            print("GeminiClient: No active WebSocket client to close.")
