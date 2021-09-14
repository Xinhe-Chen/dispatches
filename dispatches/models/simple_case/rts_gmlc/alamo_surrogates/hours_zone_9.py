import numpy as np
def f(*X):
    pmax= X[0]
    pmin= X[1]
    ramp_rate= X[2]
    min_up_time= X[3]
    min_down_time= X[4]
    marg_cst= X[5]
    no_load_cst= X[6]
    st_time_hot= X[7]
    st_time_warm= X[8]
    st_time_cold= X[9]
    st_cst_hot= X[10]
    st_cst_warm= X[11]
    st_cst_cold= X[12]
    return  1.1081228214355816064085 * pmax - 0.71042709116406732583471 * pmin - 0.26244730692955298145819 * ramp_rate + 0.24921777083595417906503E-001 * min_up_time + 0.40592190638834971250226E-001 * min_down_time - 0.15332457031267060498791 * marg_cst - 0.12361428819219352770453 * no_load_cst - 0.22942798729410945890450 * st_time_hot + 0.25169063604604816064558E-001 * st_time_warm - 0.27243778149248394637727E-001 * st_time_cold + 0.70071596460003798823024E-001 * st_cst_hot + 0.16093692208717932934370 * st_cst_warm - 0.25158479530305455362793 * pmax**2 - 0.77512843808492049024750E-001 * pmin**2 - 0.12477581265581677572030 * ramp_rate**2 + 0.46081464581555469511853E-002 * min_up_time**2 + 0.40919598217442627874352E-001 * min_down_time**2 + 0.10333098103875561213361 * marg_cst**2 - 0.25393315416196063361021E-001 * no_load_cst**2 + 0.10989160932618520505333 * st_time_hot**2 + 0.10353218708311616447215 * pmax*pmin + 0.24849757174509301549392 * pmax*ramp_rate + 0.66880368449184898413384E-002 * pmax*min_up_time + 0.20774796649846476864765E-001 * pmax*marg_cst - 0.31515917494402613530102E-001 * pmax*no_load_cst - 0.13430127307086911492284E-001 * pmax*st_time_hot - 0.51448357374328235191996E-001 * pmax*st_time_warm + 0.83095127541243818392047E-001 * pmax*st_cst_warm - 0.11173049607360696633407 * pmin*ramp_rate - 0.18286689569763581869610E-001 * pmin*marg_cst + 0.89461292642546361153499E-002 * pmin*no_load_cst + 0.39333644772488506394237E-002 * pmin*st_time_hot + 0.37152484538445309548982E-002 * pmin*st_time_warm + 0.52477603681842007604663E-003 * pmin*st_time_cold - 0.10638050605804002038401E-001 * ramp_rate*min_down_time + 0.19554184288537300451249E-001 * ramp_rate*marg_cst + 0.34062520537113921692551E-001 * ramp_rate*st_time_hot + 0.77615205519829394420483E-002 * ramp_rate*st_time_cold - 0.53630473947197582207380E-001 * ramp_rate*st_cst_hot + 0.30215400361594348110916E-001 * min_up_time*marg_cst + 0.25360413551704371476481E-001 * min_up_time*no_load_cst + 0.52051827169859368055205E-002 * min_up_time*st_time_hot + 0.10427799934920295443774E-001 * min_up_time*st_time_warm + 0.33847849275746411112920E-001 * min_up_time*st_time_cold - 0.53730370854552053638820E-001 * min_up_time*st_cst_hot + 0.85986915835019727544219E-002 * min_down_time*no_load_cst + 0.84087541149727579314899E-001 * min_down_time*st_time_hot - 0.26474289051017039359204E-001 * min_down_time*st_time_warm - 0.33788636163165665371455E-001 * min_down_time*st_cst_hot - 0.14320830002451079288051 * marg_cst*no_load_cst - 0.81857692014692146065258E-001 * marg_cst*st_time_hot - 0.36989838950267444304953E-002 * marg_cst*st_time_warm - 0.67462271620799619045727E-002 * marg_cst*st_time_cold + 0.91080206770267665983276E-001 * marg_cst*st_cst_warm - 0.66360073469894351183562E-001 * no_load_cst*st_time_hot + 0.67020141659522589408504E-002 * no_load_cst*st_time_warm - 0.27336447737889218306817E-001 * no_load_cst*st_time_cold + 0.68713183918676271066950E-001 * no_load_cst*st_cst_hot + 0.38754518027330646379180E-001 * (pmax*pmin)**2 + 0.14932485623179269154659E-001 * (pmax*ramp_rate)**2 - 0.13202790710467246504400E-001 * (pmax*marg_cst)**2 - 0.87245201004190042065600E-002 * (pmin*marg_cst)**2 + 0.10017049154628801608397E-002 * (pmin*st_time_hot)**2 - 0.64290911177954527047640E-002 * (pmin*st_time_warm)**2 - 0.51477912625935533241783E-002 * (pmin*st_time_cold)**2 + 0.23808086526116791631358E-001 * (pmin*st_cst_hot)**2 + 0.10142808391130190420748E-001 * (ramp_rate*marg_cst)**2 + 0.65216688272407693713340E-002 * (ramp_rate*no_load_cst)**2 - 0.14608605962872653338813E-001 * (ramp_rate*st_time_hot)**2 - 0.73908381887034836632555E-002 * (min_up_time*min_down_time)**2 + 0.36641781732958903644581E-002 * (min_up_time*marg_cst)**2 - 0.64140065326939192311140E-002 * (min_up_time*st_time_cold)**2 - 0.75825609872492982788117E-002 * (min_down_time*marg_cst)**2 - 0.12183641997380043400789E-001 * (min_down_time*st_time_hot)**2 - 0.75003933490629106847769E-002 * (min_down_time*st_cst_warm)**2 - 0.29224201207981446298811E-001 * (marg_cst*no_load_cst)**2 - 0.46406340137677012935846E-001 * (marg_cst*st_time_hot)**2 - 0.10789930120848563455116E-001 * (marg_cst*st_time_warm)**2 - 0.12599498946926385167799E-001 * (marg_cst*st_time_cold)**2 + 0.52016598210673968549766E-001 * (marg_cst*st_cst_hot)**2 + 0.37651400238560492339523E-001 * (no_load_cst*st_time_hot)**2 - 0.99306085773719580261920E-002 * (no_load_cst*st_time_warm)**2 