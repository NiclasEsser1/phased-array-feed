#!/usr/bin/env python
import sys
import signal
import tornado
from tornado.gen import coroutine
from katcp import Sensor, AsyncDeviceServer
from katcp.kattypes import request, return_reply, Float, Str
from optparse import OptionParser

class PafBackendController(AsyncDeviceServer):
    VERSION_INFO = ("paf-backend-controller-api", 0, 1)
    BUILD_INFO = ("paf-backend-controller-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port):
        super(PafBackendController, self).__init__(ip, port)

    def setup_sensors(self):
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of PafBackendController",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)

    def start(self):
        super(PafBackendController, self).start()

    @request(Str())
    @return_reply(Str())
    def request_echo(self, req, message):
        """
        @brief      A request that echos a message
        """
        return ("ok", message)


@coroutine
def on_shutdown(ioloop, server):
    yield server.stop()
    ioloop.stop()


def main(host, port):
    server = PafBackendController(host, port)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))
    def start_and_display():
        server.start()
        print "Server started at", server.bind_address

    ioloop = tornado.ioloop.IOLoop.current()
    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST',
                      help='attach to server HOST (default="" - localhost)')
    parser.add_option('-p', '--port', dest='port', type=int, default=17107, metavar='N',
                      help='attach to server port N (default=17107)')
    (opts, args) = parser.parse_args()
    sys.argv = sys.argv[:1]
    
    main(opts.host, opts.port)
