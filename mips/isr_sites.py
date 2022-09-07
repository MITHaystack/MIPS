"""
    isr_sites.py

    A dictionary containing sites by name as latitude, longitude, altitude. A
    helper function to enable building arrays from the dictionary.

"""

import numpy
import traceback

global isr_sites
global existing_radars
global concept_radars

# key : (type, az_rotation, el_tilt, steering_mask, freq, tx_gain, rx_gain, tx_power, duty, tsys_type, xtra_tsys, notes)
# note steering mask coordinates are az_start,az_stop,el_start,el_stop with az +/- 180 around boresite and el 0 to 90.0 relative to horizon
#
# The steering mask is in az el coordinates for non-planar arrays [az_min, az_max, el_min, el_max]
# planar arrays are masked in theta phi coordinates with boresite tilt [theta_min, theta_max, phi_min, phi_max, theta_tilt, phi_tilt]
#  theta goes from 0 to 180 degrees, phi from 0 to 360 degrees
existing_radars = {
             'millstone_misa'   : ('dish',0.0,0.0,[0.0,360.0,4.5,90.0],440.0e6,42.5,42.5,2.5e6,0.06,'fixed_high',50.0,'MISA'),
             'millstone_zenith' : ('dish',0.0,0.0,[188.0,189.0,88.0,89.0],440.0e6,45.5,45.5,2.5e6,0.06,'fixed_high',50.0,'Zenith'),
             'jicamarca'        : ('planar_array',0.0,0.0,[0.0,360.0,0.0,3.0],49.9e6,45.0,45.0,4.0e6,0.06,'fixed_high',500.0,'imaging radar'),
             'arecibo'          : ('dish',0.0,0.0,[0.0,360.0,70.0,90.0],430.0e6,58.5,58.5,1.6e6,0.06,'fixed_high',0.0,'dual beam ; limited TX bandwidth'),
             'sondrestrom'      : ('dish',0.0,0.0,[0.0,360.0,20.0,90.0],1290e6,49.9,49.9,4.0e6,0.03,'fixed_high',30.0,''),
             'poker_flat'       : ('planar_array',16.0,15.0,[0.0,360.0,0.0,38.0],449.0e6,42.0,42.0,1.8e6,0.1,'fixed_high',50.0,'AMISR'),
# poker 16.0az , 15.0el
             'resolute_n'       : ('planar_array',35.0,26.0,[0.0,360.0,0.0,38.0],449.0e6,42.0,42.0,1.8e6,0.1,'fixed_high',50.0,'AMISR'),
             'resolute_c'       : ('planar_array',206.0,35.0,[0.0,360.0,0.0,38.0],449.0e6,42.0,42.0,1.8e6,0.1,'fixed_high',50.0,'AMISR'),
             'tromso'           : ('dish',0.0,0.0,[0.0,360.0,30.0,90.0],226.7e6,46.0,46.0,1.6e6,0.125,'fixed_medium',200.0,'limited steering'),
             'kilpisjarvi'      : ('planar_array',0.0,0.0,[0.0,360.0,0.0,40.0],226.7e6,0.0,32.0,0.0,0.0,'fixed_low',0.0,'matched to VHF TX'),
             'sodonkyla'        : ('dish',0.0,0.0,[0.0,360.0,0.0,90.0],226.7e6,0.0,34.5,0.0,0.0,'fixed_high',30.0,'bistatic VHF RX only'),
             'kiruna'           : ('dish',0.0,0.0,[0.0,360.0,0.0,90.0],226.7e6,0.0,34.5,0.0,0.0,'fixed_high',30.0,'bistatic VHF RX only'),
             'svalbard'         : ('dish',0.0,0.0,[0.0,360.0,2.5,90.0],500.0e6,44.8,44.8,1.0e6,0.25,'fixed_high',0.0,''),
             'svalbard_fa'      : ('dish',0.0,0.0,[0.0,360.0,81.0,82.0],500.0e6,42.5,42.5,1.0e6,0.25,'fixed_high',0.0,'field aligned fixed dish'),
             'kharkov'          : ('dish',0.0,0.0,[0.0,360.0,89.0,90.0],158.0e6,42.0,42.0,2.0e6,0.05,'fixed_high',50.0,'fixed zenith dish'),
             'irkutsk'          : ('dish',0.0,0.0,[-15.0,15.0,60.0,90.0],158.0e6,35.0,35.0,3.2e6,0.02,'fixed_high',700.0,'parabolic trough antenna'),
             'mu'               : ('planar_array',0.0,0.0,[0.0,360.0,0.0,40.0],46.5e6,34.0,34.0,1.0e6,0.05,'fixed_high',200.0,'MST array'),
             'qujing'           : ('dish',0.0,0.0,[0.0,360.0,2.5,90.0],500e6,40.5,40.5,2.0e6,0.05,'fixed_high',0.0,'approximate parameters'),
}

concept_radars = {
             'millstone_misa_upgrade': ('dish',0.0,0.0,[0.0,360.0,4.5,90.0],440.0e6,43.281,43.281,2.0e6,0.06,'fixed_zero',63.0578,'MISA Upgrade - 2.0 MW, fixed effective Tsys=85K'),
             'millstone_zenith_upgrade' : ('dish',0.0,0.0,[188.0,189.0,88.0,89.0],440.0e6,47.78,47.78,2.0e6,0.06,'fixed_zero',78.0578,'Zenith Upgrade - 2.0 MW, fixed effective Tsys=100K'),
             'amisr_full_notilt'     : ('planar_array',0.0,0.0,[0.0,360.0,0.0,38.0],449.0e6,42.0,42.0,2.0e6,0.1,'fixed_high',50.0,'AMISR'),
             'amisr_typical_notilt'  : ('planar_array',0.0,0.0,[0.0,360.0,0.0,38.0],449.0e6,40.9,41.5,1.6e6,0.1,'fixed_high',50.0,'AMISR'),
             'heliosphere_hf'        : ('planar_array',0.0,0.0,[0.0,360.0,0.0,60.0],30.0e6,41.5,41.5,5.077e6,1.0,'fixed_medium',0.0,'4062 TX, 6093 RX elements'),
             'eiscat3d_tx'           : ('planar_array',0.0,0.0,[0.0,360.0,0.0,60.0],223.0e6,43.0,43.0,5.0e6,0.2,'fixed_high',50.0,'notional E3D T/R site'),
             'eiscat3d_rx'           : ('planar_array',0.0,0.0,[0.0,360.0,0.0,60.0],223.0e6,43.0,43.0,0.0e6,0.0,'fixed_low',0.0,'notional E3D RX site'),
}


# key : (latitude, longitude, altitude (meters), site elevation mask, description, notes)
# note: set altitude to an approximate round value when uncertain (e.g. 0.0 near coast, 100.0 interior, etc)
isr_sites = {# Existing IS Radar sites
             'millstone'  : (42.6195, -71.49173, 146.0, 2.5, 'Millstone Hill UHF radar site','existing 46 meter IS radar'),
             'jicamarca'  : (-11.95, -76.87, 500.0, 30.0,'Jicamarca radar site','existing 300 meter HF IS radar'),
             'arecibo'    : (18.34417,-66.75278, 323.0, 45.0,'Arecibo radar site','existing 300 meter UHF IS radar and HF heater'),
             'sondrestrom': (66.978,-50.949, 177.0,30.0,'Sondrestrom radar site','existing L-band IS radar'),
             'poker_flat' : (65.1260,-147.4789,750.0,20.0, 'Poker flat rocket range and radar','existing UHF AMISR IS radar'),
             'resolute'   : (74.72955,-94.90576,145.0,5.0,'Resolute bay radar site','Dual existing UHF AMISR IS radars'),
             'altair'     : (9.395380, 167.479181,2.0,5.0,'Altair radar site','Existing VHF and UHF IS capable radar'),
             'tromso'     : (69.5864,19.2272,86.28,30.0, 'EISCAT VHF radar site','VHF radar and HF heater'),
             'sodonkyla'  : (67.36361,26.62694,197.03,20.0,'EISCAT VHF receiver site','VHF receive only dish'),
             'kiruna'     : (67.86056,20.43528,417.62,20.0,'EISCAT VHF receiver site','VHF receive only dish'),
             'kilpisjarvi': (69.097253, 20.753892,565.8,20.0,'KAIRA LOFAR receiver site','Broadband receive array'),
             'svalbard'   : (78.15305,16.02889,445.0,30.0,'EISCAT Svalbard radar site','existing UHF IS radar'),
             'kharkov'    : (50.0,36.2,154.83,5.0,'Kharkov IS radar site','existing VHF IS radar and HF heater ; coordinates not accurate'),
             'irkutsk'    : (52.878012,103.265857,0.502,5.0,'Irkutsk IS radar site','existing VHF IS radar'),
             'mu'         : (34.854083, 136.105517,253.0,5.0,'Mu IS radar site','existing HF IS radar'),
             'qujing'     : (25.6,103.8,1873.0,5.0,'Qujing IS radar site','existing UHF IS radar ; coordinates not accurate'),
             'sanya'     : (18.3,109.6,30.0,5.0, 'Sanya IS radar site','planned UHF phased array radar ; coordinates not accurate'),

             # EISCAT 3D sites
             'skibotn'     : (69.340,20.313,10.0,15.0, 'planned E3D IS radar site','planned VHF phased array radar TX / RX site'),
             'kaiseniemi'  : (68.26711,19.44805,100.0,15.0, 'planned E3D IS radar site','planned VHF phased array radar RX site'),
             'karesuvanto' : (68.463,22.458,337.0,15.0, 'planned E3D IS radar site','planned VHF phased array radar RX site'),
             'palojoensuu' : (68.296716,23.079827,280.0,5.0, 'planned E3D IS radar site','possible VHF phased array radar RX site'),
             'kaitum'      : (67.637338,19.739984,508.0,5.0,'planned E3D IS radar site','possible VHF phased array radar RX site ; middle of nowhere?'),
             'andoya'      : (68.967142,15.872391,10.0,20.0,'planned E3D IS radar site','possible VHF phased array radar RX site ; approximate'),
}

def build_site_lists(sites):
    lats = []
    lons = []
    alts = []
    masks = []
    #print sites
    for s in sites:
        #print 'load %s' % (s)
        try:
            d = isr_sites[s]
            lats.append(d[0])
            lons.append(d[1])
            alts.append(d[2])
            masks.append(d[3])
        except:
            print("unknown site %s in build_site_lists, ignored." % (s))
            continue

    return (lats,lons,alts,masks)


def build_radar_lists(radars):
    rtype = []
    boresite=[]
    smask = []
    freq = []
    txg = []
    rxg = []
    txpwr = []
    duty = []
    tsys_type = []
    xtra_tsys = []
    notes = []

    #print radars
    for r in radars:
        #print 'load %s' % (r)
        try:
            try:
                d = existing_radars[r]
            except:
                try:
                    d = concept_radars[r]
                except:
                    traceback.print_exc()
                    print("unknown radar system type %s in build_radar_lists" % (r))

            print(d)
            rtype.append(d[0])
            boresite.append(numpy.array([d[1],d[2]]))
            smask.append(d[3])
            freq.append(d[4])
            txg.append(d[5])
            rxg.append(d[6])
            txpwr.append(d[7])
            duty.append(d[8])
            tsys_type.append(d[9])
            xtra_tsys.append(d[10])
            notes.append(d[1])

        except:
            traceback.print_exc()
            print("problem with radar system type %s in build_radar_lists" % (r))
            continue

    return (rtype,boresite,smask,freq,txg,rxg,txpwr,duty,tsys_type,xtra_tsys,notes)
