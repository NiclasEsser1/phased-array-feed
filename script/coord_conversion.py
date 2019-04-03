#!/usr/bin/env python
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as units
from astropy.utils import iers
import numpy as np
import sympy as sym

def _rotation_matrix(angle, d):
    directions = {
        "x":[1,0,0],
        "y":[0,1,0],
        "z":[0,0,1]
    }
    direction = np.array(directions[d])
    sina = sym.sin(angle)
    cosa = sym.cos(angle)                                                            
    R = sym.Matrix([[cosa,0,0],[0,cosa,0],[0,0,cosa]])
    R += sym.Matrix(np.outer(direction, direction)) * (1 - cosa)
    direction = sym.Matrix(direction)
    direction *= sina
    R += sym.Matrix([[ 0,           -direction[2],  direction[1]],
                     [ direction[2], 0,             -direction[0]],
                     [-direction[1], direction[0],   0]])
    return R

def _coord_convertion(time, ra, dec, beam_alt_d, beam_az_d, beam_id):
    #log.error(Time.now())
    site = EarthLocation(lat=50.524722*units.deg,
                        lon=6.883611*units.deg,
                         height=0.0*units.m)
    #log.error(Time.now())
    #sc = SkyCoord(float(ra), float(dec), unit='deg', frame='icrs',equinox="J2000")
    sc = SkyCoord(np.array(float(ra)), np.array(float(dec)), unit='deg',frame='icrs',equinox="J2000")
    #log.error(Time.now())
    aa_frame = AltAz(obstime = time, location=site)
    #print Time.now()
    beamzero_altaz = sc.transform_to(aa_frame)
    #print Time.now()
    
    daz,delv = sym.symbols('Azd Elvd')
    #log.error(Time.now())
    azel_vec = _position_vector(daz,delv)
    #log.error(Time.now())
    az, elv = sym.symbols('Az Elv')
    #log.error(Time.now())
    R = _rotation_matrix(az,"z")*_rotation_matrix(elv,"y")
    #log.error(Time.now())
    dp = R*azel_vec
    #log.error(Time.now())
    
    final_azelv = dp.subs({az:beamzero_altaz.az.radian,
                           elv:beamzero_altaz.alt.radian,
                           daz:(beam_az_d[beam_id]/180.0)*np.pi,
                           delv:(beam_alt_d[beam_id]/180.0)*np.pi})
    #log.error(Time.now())
    a, b, c = np.array(final_azelv).astype("float64")
    #log.error(Time.now())
    final_elv = np.arcsin(c)
    #log.error(Time.now())
    final_az =(np.arctan2(b,a))
    #log.error(Time.now())
    beam_pos = SkyCoord(final_az,
                        np.abs(final_elv),
                        unit='radian',
                        frame='altaz',
                        location=site,
                        obstime=time)
    #log.error(Time.now())
    beam_ra, beam_dec = beam_pos.icrs.ra.to_string(unit=units.hourangle, sep=":"), \
                        beam_pos.icrs.dec.to_string(unit=units.degree, sep=":")
    #log.error(Time.now())
    
    return beam_ra[0], beam_dec[0]

def _position_vector(a,b):
    return sym.Matrix([[sym.cos(b)*sym.cos(a)],
                       [sym.cos(b)*sym.sin(a)],
                       [sym.sin(b)]])

def _refresh_iers():
    '''
    The IERS data are managed via a instances of the IERS_Auto class. 
    These instances are created internally within the relevant time and coordinate objects during transformations.
    If the astropy data cache does not have the required IERS data file then astropy will request the file from the IERS service.
    This will occur the first time such a transform is done for a new setup or on a new machine. 
    '''
    t = Time('2019:001')
    t.ut1
    iers.conf.auto_max_age = None
    iers.conf.auto_download = False

if __name__ == "__main__":
    beam_alt_d = [0, -0.1, -0.2, -0.3, -0.1, -0.2, -0.3,  0.1,  0.2,  0.3, -0.11, -0.21, -0.31, -0.11, -0.21, -0.31,  0.12,  0.22,  0.32, -0.12, -0.22, -0.32, -0.12, -0.22, -0.32,  0.12,  0.22,  0.32, 0.13, 0.23,  0.33, -0.13, -0.23, -0.33, -0.13, -0.23, -0.33,  0.13,  0.23]
    beam_az_d = [0, -0.1, -0.2, -0.3,  0.1,  0.2,  0.3, -0.1, -0.2, -0.3, -0.11, -0.21, -0.31,  0.11,  0.21,  0.31, -0.12, -0.22, -0.32, -0.12, -0.22, -0.32,  0.12,  0.22,  0.32, -0.12, -0.22, -0.32, -0.13, -0.23, -0.33, -0.13, -0.23, -0.33,  0.13,  0.23,  0.33, -0.13, -0.23]

    ra  = 190.3
    dec = 80.1
    
    utc_start_process = Time(Time.now(), format='isot', scale='utc').value

    print Time.now()
    
    _refresh_iers()
    print Time.now()
    _coord_convertion(utc_start_process, ra, dec,
                      beam_alt_d, beam_az_d, 10)
    print Time.now()
