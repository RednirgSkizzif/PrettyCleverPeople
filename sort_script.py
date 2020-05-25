from glob import glob
from pprint import pprint 
files = glob("custom*/*.hdf5")
struct = [ (x.split('/')[0],int( float( x.split('/')[-1][-11:-5] ) *100000) ) for x in files ]
struct.sort(reverse=True,key= lambda i: int(i[1]))
pprint(struct)
