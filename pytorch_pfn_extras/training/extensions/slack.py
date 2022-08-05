import getpass
import os
import json
import urllib.request
import shlex
import sys
import socket
import warnings

from typing import Any, Callable, List, Optional, Union, TYPE_CHECKING

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike


try:
    import slack_sdk
    _slack_sdk_available = True
except ImportError:
    _slack_sdk_available = False


_identity = f'{getpass.getuser()}@{socket.gethostname()} [PID {os.getpid()}]'


def _default_start_msg(*args, **kwargs) -> str:
    return f'''**Training started! {_identity}**
Command: `{shlex.quote(' '.join(sys.argv))}`
'''


_default_end_msg = f'**Training finished! {_identity}**'


class Slack(extension.Extension):
    """An extension to communicate with Slack.

    This extension receives a ``msg`` argument that contains
    placeholders that will be populated with ``manager`` or ``manager.observation``
    fields, and custom objects such as ``context_object``.

    For example. `msg = "Loss {loss} for iteration {.iteration}"`
    will be populated as `msg.format(manager, context_object, **observation)`
    retrieving the ``loss`` value from the ``observation`` and ``.iteration`` from
    the manager object. Instead of string, a callable object taking the
    ``ExtensionsManager``, ``context_object`` and the observation dictionary can
    be used instead. The same rules apply for the ``filenames_template`` argument.

    Args:
        channel_id (str): The channel where messages or files will be sent.
        msg (str or callable): Template for sending the message.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string. In both cases, `manager`, the current
            observation dict and the context object are used to look up
            values of interest.
        start_msg (str or callable): Template for sending a message
            at the beggining of the experiment. If ``None``, the default
            start message will be sent. To avoid sending a message use
            ``''``.
            See ``msg`` for format.
        end_msg (str or callable): Template for sending a message
            at the completion of the experiment. If ``None``, the default
            end message will be sent. To avoid sending a message use
            See ``msg`` for format.
        filenames(list of str or callable): list of files that will
            be uploaded to slack, these are string templates that can take
            values in the same way as ``msg``. Optional.
        thread (bool): If subsequent calls of this extension should be
            posted as a thread of the original message or not.
            Default is ``False``.
        context_object (object): Custom object that contains data used to
            format the text or the filenames. Optional, default is ``None``.
        token (str): Token for the slack api, if ``None`` the environment
            variable ``SLACK_TOKEN`` will be used. Ignored if ``client`` is
            supplied. Optional, default is ``None``.
        client (slack_sdk.WebClient): In case that there is an already created
            slack client in the application, allows to directly use it.
            Optional, default is ``None``
    """

    trigger: 'TriggerLike' = (1, 'epoch')

    def __init__(
        self,
        channel_id: str,
        msg: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        *,
        start_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = None,
        end_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = None,
        filenames: Optional[
            List[
                Union[
                    str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
                ]
            ]
        ] = None,
        thread: bool = False,
        context_object: Optional[object] = None,
        token: Optional[str] = None,
        client: Optional[Any] = None,  # slack_sdk.WebClient, Any to avoid mypy errors
    ) -> None:
        if not _slack_sdk_available:
            warnings.warn(
                '`slack_api` package is unavailable. '
                'Slack will do nothing.')
            return

        self._client = client
        if client is None:
            if token is None:
                token = os.environ.get('SLACK_TOKEN', None)
            if token is None:
                raise RuntimeError(
                    '`token` is needed for communicating with slack')
            self._client = slack_sdk.WebClient(token=token)

        # values in current observation or log report to send to slack
        self._msg = msg
        if start_msg is None:
            self._start_msg = _default_start_msg
        else:
            self._start_msg = start_msg
        if end_msg is None:
            self._end_msg = _default_end_msg
        else:
            self._end_msg = end_msg
        if filenames is None:
            filenames = []
        self._filenames = filenames
        self._context = context_object
        self._channel_id = channel_id
        self._thread = thread
        self._ts = None

    def _send_message(
        self,
        manager: ExtensionsManagerProtocol,
        text: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]
    ) -> None:
        if not _slack_sdk_available:
            return

        observation = manager.observation
        if callable(text):
            text = text(manager, self._context, observation)
        else:
            print(text)
            text = text.format(
                manager=manager, context=self._context, **observation)

        assert self._client is not None

        ts = None
        if self._thread:
            ts = self._ts
        response = self._client.chat_postMessage(
            channel=self._channel_id,
            text=text,
            thread_ts=ts,
        )
        assert response.get("ok")  # type: ignore[no-untyped-call]
        if self._thread and ts is None:
            ts = response.get("ts")  # type: ignore[no-untyped-call]
            self._ts = ts
        for filename in self._filenames:
            if callable(filename):
                filename = filename(manager, self._context, observation)
            else:
                filename = filename.format(
                    manager=manager, context=self._context, **observation)
            response = self._client.files_upload(
                channels=self._channel_id,
                file=filename,
                thread_ts=ts,
            )
            assert response.get("ok")  # type: ignore[no-untyped-call]

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._start_msg != '':
            self._send_message(manager, self._start_msg)

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._end_msg != '':
            self._send_message(manager, self._end_msg)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available:
            self._send_message(manager, self._msg)


class SlackWebhook(extension.Extension):
    """An extension to communicate with Slack.

    This extension receives a ``msg`` argument that contains
    placeholders that will be populated with ``manager`` or ``manager.observation``
    fields, and custom objects such as ``context_object``.

    For example. `msg = "Loss {loss} for iteration {.iteration}"`
    will be populated as `msg.format(manager, context_object, **observation)`
    retrieving the ``loss`` value from the ``observation`` and ``.iteration`` from
    the manager object. Instead of string, a callable object taking the
    ``ExtensionsManager``, ``context_object`` and the observation dictionary can
    be used instead. The same rules apply for the ``filenames_template`` argument.

    Args:
        webhook_url (str): Incoming webhook URL to send messages.
        msg (str or callable): Template for sending the message.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string. In both cases, `manager`, the current
            observation dict and the context object are used to look up
            values of interest.
        start_msg (str or callable): Template for sending a message
            at the beggining of the experiment.
            See ``msg`` for format.
        end_msg (str or callable): Template for sending a message
            at the completion of the experiment.
            See ``msg`` for format.
        context_object (object): Custom object that contains data used to
            format the text or the filenames. Optional, default is ``None``.
    """

    trigger: 'TriggerLike' = (1, 'epoch')

    def __init__(
        self,
        webhook_url: str,
        msg: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        *,
        start_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = None,
        end_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = None,
        context_object: Optional[object] = None,
    ) -> None:

        self._webhook_url = webhook_url
        # values in current observation or log report to send to slack
        self._msg = msg
        if start_msg is None:
            self._start_msg = _default_start_msg
        else:
            self._start_msg = start_msg
        if end_msg is None:
            self._end_msg = _default_end_msg
        else:
            self._end_msg = end_msg
        self._context = context_object

    def _send_message(
        self,
        manager: ExtensionsManagerProtocol,
        text: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]
    ) -> None:
        observation = manager.observation
        if callable(text):
            text = text(manager, self._context, observation)
        else:
            text = text.format(
                manager=manager, context=self._context, **observation)

        if self._webhook_url:
            payload = json.dumps({'text': text}).encode('utf-8')
            request_headers = {'Content-Type': 'application/json; charset=utf-8'}
            # res = requests.post(self._webhook_url, json.dumps(payload))
            request = urllib.request.Request(
                url=self._webhook_url,
                data=payload,
                method='POST',
                headers=request_headers
            )
            urllib.request.urlopen(request)

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._start_msg != '':
            self._send_message(manager, self._start_msg)

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._end_msg != '':
            self._send_message(manager, self._end_msg)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        self._send_message(manager, self._msg)
