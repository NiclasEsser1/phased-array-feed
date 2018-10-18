#!/usr/bin/env python

import socket, struct, json, ephem, datetime, pytz

def getDUTCDt(dt=None):
    """
    Get the DUTC value in seconds that applied at the given (datetime)
    timestamp.
    """
    dt = dt or utc_now()
    for i in LEAPSECONDS:
        if i[0] < dt:
            return i[2]
    # Different system used before then
    return 0

# Leap seconds definition, [UTC datetime, MJD, DUTC]
LEAPSECONDS = [
    [datetime.datetime(2017, 1, 1, tzinfo=pytz.utc), 57754, 37],
    [datetime.datetime(2015, 7, 1, tzinfo=pytz.utc), 57205, 36],
    [datetime.datetime(2012, 7, 1, tzinfo=pytz.utc), 56109, 35],
    [datetime.datetime(2009, 1, 1, tzinfo=pytz.utc), 54832, 34],
    [datetime.datetime(2006, 1, 1, tzinfo=pytz.utc), 53736, 33],
    [datetime.datetime(1999, 1, 1, tzinfo=pytz.utc), 51179, 32],
    [datetime.datetime(1997, 7, 1, tzinfo=pytz.utc), 50630, 31],
    [datetime.datetime(1996, 1, 1, tzinfo=pytz.utc), 50083, 30],
    [datetime.datetime(1994, 7, 1, tzinfo=pytz.utc), 49534, 29],
    [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 49169, 28],
    [datetime.datetime(1992, 7, 1, tzinfo=pytz.utc), 48804, 27],
    [datetime.datetime(1991, 1, 1, tzinfo=pytz.utc), 48257, 26],
    [datetime.datetime(1990, 1, 1, tzinfo=pytz.utc), 47892, 25],
    [datetime.datetime(1988, 1, 1, tzinfo=pytz.utc), 47161, 24],
    [datetime.datetime(1985, 7, 1, tzinfo=pytz.utc), 46247, 23],
    [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 45516, 22],
    [datetime.datetime(1982, 7, 1, tzinfo=pytz.utc), 45151, 21],
    [datetime.datetime(1981, 7, 1, tzinfo=pytz.utc), 44786, 20],
    [datetime.datetime(1980, 1, 1, tzinfo=pytz.utc), 44239, 19],
    [datetime.datetime(1979, 1, 1, tzinfo=pytz.utc), 43874, 18],
    [datetime.datetime(1978, 1, 1, tzinfo=pytz.utc), 43509, 17],
    [datetime.datetime(1977, 1, 1, tzinfo=pytz.utc), 43144, 16],
    [datetime.datetime(1976, 1, 1, tzinfo=pytz.utc), 42778, 15],
    [datetime.datetime(1975, 1, 1, tzinfo=pytz.utc), 42413, 14],
    [datetime.datetime(1974, 1, 1, tzinfo=pytz.utc), 42048, 13],
    [datetime.datetime(1973, 1, 1, tzinfo=pytz.utc), 41683, 12],
    [datetime.datetime(1972, 7, 1, tzinfo=pytz.utc), 41499, 11],
    [datetime.datetime(1972, 1, 1, tzinfo=pytz.utc), 41317, 10],
]
"# Leap seconds definition, [UTC datetime, MJD, DUTC]"

def bat2utc(bat, dutc=None):
    """
    Convert Binary Atomic Time (BAT) to UTC.  At the ATNF, BAT corresponds to
    the number of microseconds of atomic clock since MJD (1858-11-17 00:00).

    :param bat: number of microseconds of atomic clock time since
        MJD 1858-11-17 00:00.
    :type bat: long
    :returns utcDJD: UTC date and time (Dublin Julian Day as used by pyephem)
    :rtype: float

    """
    dutc = dutc or getDUTCDt()
    if type(bat) is str:
        bat = long(bat, base=16)
    utcMJDs = (bat/1000000.0)-dutc
    utcDJDs = utcMJDs-86400.0*15019.5
    utcDJD = utcDJDs/86400.0
    return utcDJD + 2415020 - 2400000.5

def utc_now():
    """
    Return current UTC as a timezone aware datetime.
    :rtype: datetime
    """
    dt=datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt

# Bind to multicast
multicast_group = '224.1.1.1'
multicast_port = 5007
server_address = ('', multicast_port)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create the socket

sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(server_address) # Bind to the server address

group = socket.inet_aton(multicast_group)
mreq = struct.pack('4sL', group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)  # Tell the operating system to add the socket to the multicast group on all interfaces.

fp = open('{:s}.observation.log'.format(utc_now().strftime("%Y-%m-%d-%H:%M:%S")), 'w')
scan_id0 = '0'
while(True):
    pkt, address = sock.recvfrom(1<<16)
    data = json.loads(pkt)

    scan_id = str(data['scan_id'])

    if not (scan_id == scan_id0):
        scan_id0 = scan_id
        
        timestamp   = str(bat2utc(str(data['timestamp'])))
        target_name = str(data['target_name'])
        ra          = str(ephem.hours(float(data['pk01']['actual_radec'][0])))
        dec         = str(ephem.degrees(float(data['pk01']['actual_radec'][1])))
        az          = str(ephem.hours(float(data['pk01']['actual_azel'][0])))
        el          = str(ephem.degrees(float(data['pk01']['actual_azel'][1])))

        fp.write(timestamp)
        fp.write('\t')
        fp.write(scan_id)
        fp.write('\t')
        fp.write(target_name)
        fp.write('\t')
        fp.write(ra)
        fp.write('\t')
        fp.write(dec)
        fp.write('\t')
        fp.write(az)
        fp.write('\t')
        fp.write(el)
        fp.write('\n')
        
fp.close()
