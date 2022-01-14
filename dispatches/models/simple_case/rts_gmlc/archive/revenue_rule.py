from revenue_surrogate import f as revenue
from revenue_surrogate import xm,xstd,zm,zstd

def revenue_rule(m):
    #scale model inputs to surrogate
    pmax = (m.pmax - xm[0])/xstd[0]
    pmin = (m.pmin - xm[1])/xstd[1]
    ramp_rate = (m.ramp_rate - xm[2])/xstd[2]
    min_up_time = (m.min_up_time - xm[3])/xstd[3]
    min_down_time = (m.min_down_time  - xm[4])/xstd[4]
    marg_cst = (m.marg_cst - xm[5])/xstd[5]
    no_load_cst = (m.no_load_cst - xm[6])/xstd[6]
    st_time_hot = (m.st_time_hot - xm[7])/xstd[7]
    st_time_warm = (m.st_time_warm - xm[8])/xstd[8]
    st_time_cold = (m.st_time_cold - xm[9])/xstd[9]
    st_cst_hot = (m.st_cst_hot - xm[10])/xstd[10]
    st_cst_warm = (m.st_cst_warm - xm[11])/xstd[11]
    st_cst_cold = (m.st_cst_cold - xm[12])/xstd[12]
    z =  1.3806723142413952487573 * pmax - 0.48776462319249686006017 * pmin - 0.91585208990741973078542 * ramp_rate + 0.55270667881385949354867E-002 * min_up_time + 0.28769120202235868265228E-002 * min_down_time + 0.22741398560451368815460E-001 * marg_cst - 0.48343155663864205429103E-001 * no_load_cst + 0.59625125367485612426499E-001 * st_time_hot - 0.34649691968405706143930E-001 * st_time_warm - 0.56604172307272875019901E-001 * st_time_cold + 0.18073217576206789675153E-001 * st_cst_hot - 0.29460416736315945401836E-001 * st_cst_warm + 0.60367280738223109970431E-001 * pmax**2 - 0.14582916733686485111221 * pmin**2 + 0.29262239394862561703281E-001 * ramp_rate**2 - 0.67032644586412625659078E-002 * min_up_time**2 - 0.44872116509954576915598E-002 * min_down_time**2 + 0.87665879064744878235160E-001 * marg_cst**2 + 0.40547371346606707331883E-001 * no_load_cst**2 - 0.20893087937275797716374 * st_time_hot**2 + 0.13942027812629390060017 * pmax*pmin - 0.89464690727712736784127E-001 * pmax*ramp_rate - 0.21870253029559520718816E-001 * pmax*marg_cst - 0.10519788095393056703841 * pmax*no_load_cst - 0.48560849553014713564369E-001 * pmax*st_time_hot + 0.14664796828387413260564E-001 * pmax*st_time_warm - 0.13104588063438780617953E-001 * pmax*st_time_cold - 0.23683060657134349241693E-001 * pmax*st_cst_hot - 0.19961441059851409152159 * pmin*ramp_rate - 0.61435682457201420958448E-001 * pmin*marg_cst + 0.35333918434005823216992E-001 * pmin*no_load_cst - 0.39509891145257779870859E-002 * pmin*st_time_hot + 0.20209176866779032799570E-001 * pmin*st_time_warm + 0.11265358314847673595893E-001 * pmin*st_time_cold - 0.91765936720995394670908E-002 * pmin*st_cst_hot - 0.73523714713492546724005E-001 * ramp_rate*marg_cst + 0.10828746966730548595415 * ramp_rate*no_load_cst + 0.79516145586863790084564E-001 * ramp_rate*st_time_hot - 0.36753176943549542565748E-001 * ramp_rate*st_time_warm + 0.10618725184491223378913E-001 * ramp_rate*st_time_cold + 0.52582111722188365487973E-001 * ramp_rate*st_cst_hot + 0.69835839194371833113517E-002 * min_up_time*marg_cst + 0.81000818246834468266959E-002 * min_up_time*no_load_cst + 0.22691643489420520660160E-002 * min_up_time*st_time_hot + 0.38243711359654537600139E-002 * min_up_time*st_time_cold - 0.14315730065729784931117E-001 * min_down_time*marg_cst + 0.16933075902694005171467E-001 * min_down_time*st_cst_hot + 0.43421334062367741846167E-001 * marg_cst*no_load_cst - 0.17928966083488179217298E-001 * marg_cst*st_time_hot + 0.67820900201865349024577E-002 * marg_cst*st_time_warm + 0.23895349255890281810200E-002 * marg_cst*st_time_cold - 0.36328056509854336764143E-001 * marg_cst*st_cst_hot + 0.72380113016918337653927E-002 * no_load_cst*st_time_hot - 0.18783711637930702864629E-001 * no_load_cst*st_time_warm + 0.14699448301491920346185E-001 * (pmax*pmin)**2 + 0.45270693739933587362856E-001 * (pmax*ramp_rate)**2 - 0.60766390650068757492419E-002 * (pmax*marg_cst)**2 - 0.30389760097675236859283E-002 * (pmax*no_load_cst)**2 + 0.16963362694008238956700E-001 * (pmax*st_time_hot)**2 - 0.41336098718837047116814E-001 * (pmin*ramp_rate)**2 - 0.20761707512037055889387E-002 * (pmin*min_down_time)**2 + 0.10743613128965441572138E-001 * (pmin*marg_cst)**2 - 0.25149921283295177676376E-002 * (pmin*st_time_hot)**2 + 0.84056355675681225514406E-002 * (pmin*st_cst_hot)**2 - 0.59843040949532522176924E-001 * (ramp_rate*marg_cst)**2 - 0.24549676436880999569334E-001 * (ramp_rate*st_time_hot)**2 + 0.95755550862903059811115E-002 * (ramp_rate*st_cst_warm)**2 - 0.51935370826119353279693E-002 * (min_up_time*marg_cst)**2 + 0.10694739849231350153902E-001 * (min_up_time*st_time_hot)**2 - 0.72213980269235627379443E-002 * (min_down_time*marg_cst)**2 + 0.80426369547061116183073E-002 * (min_down_time*st_time_hot)**2 - 0.58875821203179905249936E-001 * (marg_cst*no_load_cst)**2 + 0.12980978849774577055243 * (marg_cst*st_time_hot)**2 + 0.18076370253294195972193E-001 * (marg_cst*st_time_warm)**2 + 0.37938061005235258760226E-001 * (marg_cst*st_time_cold)**2 - 0.50596040527398494779376E-001 * (marg_cst*st_cst_hot)**2 + 0.44794885140915343194057E-002 * (pmax*pmin)**3 - 0.95288980382401671648251E-002 * (pmax*ramp_rate)**3 + 0.50861181173934136290349E-001 * (pmax*marg_cst)**3 + 0.64088117694703324381256E-002 * (pmax*no_load_cst)**3 - 0.18605358909687293167412E-001 * (pmin*ramp_rate)**3 - 0.13622333013035791554612E-001 * (pmin*marg_cst)**3 - 0.43980356585351315992782E-002 * (ramp_rate*marg_cst)**3 - 0.33007176309487675121279E-002 * (ramp_rate*no_load_cst)**3 - 0.52123437490753246961739E-002 * (ramp_rate*st_time_hot)**3 - 0.58513929225285860394323E-002 * (ramp_rate*st_cst_hot)**3 + 0.19707119529953215364415E-002 * (min_up_time*marg_cst)**3 - 0.52394971804675685711494E-001 * (marg_cst*no_load_cst)**3

    z_unscale = z*zstd + zm
    return z_unscale