import re
RULES=[
 ('GS','bank',r'^GOLDMAN SACHS|^PHILLIP STREET|^WEST BAY BDC'),
 ('MS','bank',r'^MORGAN STANLEY|^NORTH HAVEN|^T SERIES|^SL INVESTMENT|^LGAM PRIVATE CREDIT'),
 ('OVERLAND','bank_jv',r'^OVERLAND ADVANTAGE'),
 ('JEFFERIES','bd_insurer_jv',r'^JEFFERIES CREDIT|^SENIOR CREDIT INVESTMENTS'),
 ('BARINGS','insurer',r'^BARINGS'),('NUVEEN','insurer',r'^NUVEEN CHURCHILL'),('PGIM','insurer',r'^PGIM'),
 ('CRESCENT','insurer',r'^CRESCENT'),('MANULIFE','insurer',r'^MANULIFE|^JOHN HANCOCK'),('AB','insurer',r'^AB PRIVATE'),
 ('PIMCO','insurer',r'^PIMCO'),
 ('BLACKROCK','assetmgr',r'^BLACKROCK'),('FIDELITY','assetmgr',r'^FIDELITY'),('OHA','assetmgr',r'T\. ROWE PRICE OHA|^OHA SENIOR'),
 ('FRANKLIN','assetmgr',r'^FRANKLIN BSP'),('LORDABBETT','assetmgr',r'^LORD ABBETT'),('STEPSTONE','assetmgr',r'^STEPSTONE'),
 ('PALMER','assetmgr',r'^PALMER SQUARE'),('CIFC','assetmgr',r'^CIFC'),('GUGGENHEIM','assetmgr',r'^GUGGENHEIM'),('TCW','assetmgr',r'^TCW'),
 ('FIRSTEAGLE','assetmgr',r'^FIRST EAGLE'),
 ('BLACKSTONE','alt',r'^BLACKSTONE'),('BLUEOWL','alt',r'^BLUE OWL|^OWL ROCK'),('ARES','alt',r'^ARES'),('HPS','alt',r'^HPS'),
 ('APOLLO','alt',r'APOLLO|^MIDCAP FINANCIAL'),('KKR','alt',r'^FS KKR|^KKR|^FS ENERGY|^FS SPECIALTY'),('GOLUB','alt',r'^GOLUB'),
 ('SIXTHSTREET','alt',r'^SIXTH STREET'),('OAKTREE','alt',r'^OAKTREE'),('CARLYLE','alt',r'^CARLYLE'),('BAIN','alt',r'^BAIN CAPITAL'),
 ('TWINBROOK','alt',r'TWIN BROOK'),('ANTARES','alt',r'^ANTARES'),('NEWMOUNTAIN','alt',r'^NEW MOUNTAIN|^NMF SLF'),('MONROE','alt',r'^MONROE'),
 ('MSD','alt',r'^MSD INVESTMENT'),('PENNANTPARK','alt',r'^PENNANTPARK'),('AUDAX','alt',r'^AUDAX'),('STELLUS','alt',r'^STELLUS'),
 ('KAYNE','alt',r'^KAYNE'),('SILVERPOINT','alt',r'^SILVER POINT'),('STONEPOINT','alt',r'^STONE POINT'),('ONEX','alt',r'^ONEX'),
 ('SLR','alt',r'^SLR'),('MAINSTREET','alt',r'^MAIN STREET|^MSC INCOME'),('TRIPLEPOINT','alt',r'^TRIPLEPOINT'),('MUZINICH','alt',r'^MUZINICH'),
 ('INVESTCORP','alt',r'^INVESTCORP'),('VISTA','alt',r'^VISTA'),('KENNEDYLEWIS','alt',r'^KENNEDY LEWIS'),('DIAMETER','alt',r'^DIAMETER'),
 ('PROSPECT','alt',r'^PROSPECT'),('GLADSTONE','alt',r'^GLADSTONE'),('WTI','alt',r'^WTI FUND|^VENTURE LENDING'),('SILVERCAP','exclude',r'^SILVER CAPITAL HOLDINGS'),
]
def classify(name):
    n=(name or '').upper().strip()
    for fam,typ,pat in RULES:
        if re.search(pat,n): return fam,typ
    return 'OTHER:'+n[:30],'alt'
