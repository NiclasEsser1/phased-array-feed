#!/usr/bin/env python

import threading
import time
import random
import tornado
import signal
from katcp import AsyncDeviceServer
from katcp import Message ,DeviceServer, Sensor, ProtocolFlags, AsyncReply
from katcp.kattypes import (Str, Float, Timestamp, Discrete,
                            request, return_reply)
#from katcp import reply_inform
from optparse import OptionParser

#class MyServer(DeviceServer):
class MyServer(AsyncDeviceServer):

    VERSION_INFO = ("example-api", 1, 0)
    BUILD_INFO = ("example-implementation", 0, 1, "")

    # Optionally set the KATCP protocol version and features. Defaults to
    # the latest implemented version of KATCP, with all supported optional
    # features
    PROTOCOL_INFO = ProtocolFlags(5, 0, set([
        ProtocolFlags.MULTI_CLIENT,
        ProtocolFlags.MESSAGE_IDS,
    ]))

    FRUIT = [
        "apple", "banana", "pear", "kiwi",
    ]

    def setup_sensors(self):
        """Setup some server sensors."""
        self._add_result = Sensor.float("add.result",
            "Last ?add result.", "", [-10000, 10000])

        self._time_result = Sensor.timestamp("time.result",
            "Last ?time result.", "")

        self._eval_result = Sensor.string("eval.result",
            "Last ?eval result.", "")

        self._fruit_result = Sensor.discrete("fruit.result",
            "Last ?pick-fruit result.", "", self.FRUIT)
        self._device_armed = Sensor.boolean(
            "device-armed",
            description="Is the CAM server armed?",
            initial_status=Sensor.NOMINAL,
            default=False)
        self._bandwidth = Sensor.float("bandwidth",default=300)
        self._sourcename = Sensor.string("sourcename",default="none")
        self._source_ra = Sensor.string("source_RA",default=0)
	self._source_dec = Sensor.string("source_DEC",default=0)
	self._exposure_time = Sensor.float("EXP_time",default=0)


	self.add_sensor(self._sourcename)
        self.add_sensor(self._source_ra)
        self.add_sensor(self._source_dec)
        self.add_sensor(self._exposure_time)

        self.add_sensor(self._bandwidth)
        self.add_sensor(self._device_armed)
        self.add_sensor(self._add_result)
        self.add_sensor(self._time_result)
        self.add_sensor(self._eval_result)
        self.add_sensor(self._fruit_result)

        self._systemp_result = Sensor.float("add.result",
            "Last ?add result.", "", [-10000, 10000])
        self.add_sensor(self._systemp_result)


    @request()
    @return_reply(Str())
    def request_call(self, req):
        """Todo: return an msg  when recieved a message, continute to do your work, send out another reply when it is finally done"""
        @tornado.gen.coroutine
	#req.reply("processing", "effcam armed")
        def start_controller():
#            self.request_processing(req)
#           req.reply("processing", "command processing")
#           raise AsyncReply
            try:
                yield tornado.gen.sleep(0.001)
#                yield self._controller.start()
            except Exception as error:
                req.reply("fail", "Unknown error: {0}".format(str(error)))
            else:
                req.reply("processing", "effcam armed")
                self._device_armed.set_value(True)
        if self._device_armed.value():
            return ("fail", "Effcam is already armed")
        self.ioloop.add_callback(start_controller)
        raise AsyncReply

#    @request(Float())
#    @return_reply(Str())
#    def request_set_bandwdith(self, req, BW):
#	        '''settting bandwidth'''
#	        self._bandwidth.set_value(BW)
#	return ("ok", self._bandwidth.value())

    @request()
    @return_reply(Str(),Float(),Float(),Float(),Float())
    def request_status_config_current(self, req):
        """Observation details"""
    #    return ("ok",name, ra, dec, time,self._bandwidth.value())
	req.inform("processing","processing command")
	req.reply("ok",self._sourcename.value(), self._source_ra.value(),self._source_dec.value(),self._exposure_time.value(),self._bandwidth.value())	
	raise AsyncReply
        #return ("ok",self._sourcename.value(), self._source_ra.value(),self._source_dec.value(),self._exposure_time.value(),self._bandwidth.value())

    @request(Str(),Float(),Float(),Float(),Float())
    @return_reply(Str(),Float(),Float(),Float(),Float())
    def request_status_config(self, req, name, ra, dec, time, bw):
        """Observation details"""
	self._sourcename.set_value(name)
        self._source_ra.set_value(ra)
        self._source_dec.set_value(dec)
        self._exposure_time.set_value(time)
        self._bandwidth.set_value(bw)
	req.inform("processing",self._sourcename.value(), self._source_ra.value(),self._source_dec.value(),self._exposure_time.value(),self._bandwidth.value())
 #       return ("processing",self._sourcename.value(), self._source_ra.value(),self._source_dec.value(),self._exposure_time.value(),self._bandwidth.value())
	req.reply("ok", "finsihed config")
	raise AsyncReply
    @request()
    @return_reply(Str())
    def request_status_backend(self, req):
        """Return the state of the Backend Armed/Disarmed"""
 	req.inform("processing","processing command")
        req.reply("ok", self._device_armed.value())
	raise AsyncReply
#        return ("ok", self._device_armed.value())
    @request(Float())
    @return_reply()
    def request_long_action(self, req, t):
        """submit a long action command for testing using coroutine"""
        @tornado.gen.coroutine
        def wait():
            yield tornado.gen.sleep(t)
            req.reply("slept for",t,"second")
        self.ioloop.add_callback(wait)
        raise AsyncReply

    @request(Float(), Float())
    @return_reply(Str())
    def request_radec(self, req, ra, dec):
	"""testing to read in the RA DEC fomr a client"""
        #test=ra+dec
        self.ra=ra
        self.dec=dec
#        pass
        #return ("ok",self.ra,self.dec)
        #return ("ok","RA=%f,DEC=%f"%(self.ra,self.dec))
        return ("ok","%f %f"%(self.ra,self.dec))

    @request(Float(), Float())
    @return_reply(Float())
    def request_add(self, req, x, y):
        """Add two numbers"""
        r = x + y
        self._add_result.set_value(r)
        return ("ok", r)

    @request()
    @return_reply(Str())
    def request_arm(self, req):
        """Arm the controller"""
        @tornado.gen.coroutine
	#print "testing"
#	self.processing(self)
        def start_controller():
	    req.inform("processing","command processing")
            try:
                yield tornado.gen.sleep(10)
#                yield self._controller.start()
            except Exception as error:
                req.reply("fail", "Unknown error: {0}".format(str(error)))
            else:
                req.reply("ok", "effcam armed")
                self._device_armed.set_value(True)
        if self._device_armed.value():
            return ("fail", "Effcam is already armed")
        self.ioloop.add_callback(start_controller)
        raise AsyncReply

#    @request()
#    @return_reply(Str())
#    def request_processing(self,req):
#        """return a msg to the client"""
#        @tornado.gen.coroutine
#	print "i am here"
#	req.reply("processing", "command processing")
#        def start_controller():
#            try:
#                yield tornado.gen.sleep(0.001)
#                yield self._controller.start()
#            except Exception as error:
#                req.reply("fail", "Unknown error: {0}".format(str(error)))
#            else:
#                req.reply("ok", "effcam armed")
#                self._device_armed.set_value(True)
#        if self._device_armed.value():
#            return ("fail", "Effcam is already armed")
#        self.ioloop.add_callback(start_controller)
#        raise AsyncReply
    @request()
    @return_reply(Str())
    def request_disarm(self, req):
        """disarm the controller"""
        @tornado.gen.coroutine
        #@coroutine
        def stop_controller():
            req.inform("processing","processing command")
            try:
                yield tornado.gen.sleep(10)
                #yield self._controller.stop()
            except Exception as error:
                req.reply("fail", "Unknown error: {0}".format(str(error)))
            else:
                req.reply("ok", "effcam disarmed")
                self._device_armed.set_value(False)
        if self._device_armed.value()==False:
            return ("fail", "Effcam is already disarmed")
        self.ioloop.add_callback(stop_controller)
        raise AsyncReply


    @request()
    @return_reply(Str())
    def request_add_new_sensor(self, req):
        """add a new bw sensor, just for testing"""
        #req.inform("processing","processing command")
        self._newsensor = Sensor.float("newsensor",default=300)
        self.add_sensor(self._newsensor)
        self._newsensor.set_value(300)
        self.mass_inform(Message.inform('interface-changed', 'processing command'))
        #return ("ok", self._newsensor.value())
        req.reply("ok", self._newsensor.value())
        raise AsyncReply
    @request()
    @return_reply(Str())
    def request_remove_new_sensor(self, req):
        """add a new bw sensor, just for testing"""
        #self._newsensor = Sensor.float("newsensor",default=300)
        self.remove_sensor(self._newsensor)
        #self._newsensor.set_value(300)
        self.mass_inform(Message.inform('interface-changed'))
        return ("ok", "sensor_deleted")


    @request()
    @return_reply(Str())
    def request_status_temp(self, req):
        """Return the current temp"""
        #r = time.time()
	t = "36"
        #self._time_result.set_value(r)
        return ("ok", t)


    @request()
    @return_reply(Timestamp())
    def request_status_time(self, req):
        """Return the current time in seconds since the Unix Epoch."""
	req.inform("processing","processing command")
        r = time.time()
        self._time_result.set_value(r)
	req.reply("ok", r)
	raise AsyncReply
        #return ("ok", r)

    @request()
    @return_reply(Timestamp(),Str())
    def request_status_time_and_temp(self, req):
        """Return the current time in seconds since the Unix Epoch."""
        req.inform("processing","processing command")
        r = time.time()
        self._time_result.set_value(r)
	t = "36"
	req.reply("ok", r,t)
        raise AsyncReply
    #    return ("ok", r,t)

    @request(Str())
    @return_reply(Str())
    def request_eval(self, req, expression):
        """Evaluate a Python expression."""
        r = str(eval(expression))
        self._eval_result.set_value(r)
        return ("ok", r)

    @request()
    @return_reply(Discrete(FRUIT))
    def request_pick_fruit(self, req):
        """Pick a random fruit."""
        r = random.choice(self.FRUIT + [None])
        if r is None:
            return ("fail", "No fruit.")
        delay = random.randrange(1,5)
        req.inform("Picking will take %d seconds" % delay)

        def pick_handler():
            self._fruit_result.set_value(r)
            req.reply("ok", r)

        self.ioloop.add_callback(
          self.ioloop.call_later, delay, pick_handler)

        raise AsyncReply

    def request_raw_reverse(self, req, msg):
        """
        A raw request handler to demonstrate the calling convention if
        @request decoraters are not used. Reverses the message arguments.
        """
        # msg is a katcp.Message.request object
        reversed_args = msg.arguments[::-1]
        # req.make_reply() makes a katcp.Message.reply using the correct request
        # name and message ID
        return req.make_reply('ok', *reversed_args)
@tornado.gen.coroutine
def on_shutdown(ioloop, server):
    print('Shutting down')
    yield server.stop()
    ioloop.stop()
if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST',
                      help='attach to server HOST (default="" - localhost)')
    parser.add_option('-p', '--port', dest='port', type=int, default=17107, metavar='N',
                      help='attach to server port N (default=17107)')
    (opts, args) = parser.parse_args()
    
    ioloop = tornado.ioloop.IOLoop.current()
    server = MyServer(opts.host, opts.port)
    print "Server started at", server.bind_address
    # Hook up to SIGINT so that ctrl-C results in a clean shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))
    ioloop.add_callback(server.start)
    ioloop.start()

#if __name__ == "__main__":
#
#    server = MyServer(server_host, server_port)
#    server.start()
#    print "server running"
#    server.join()
