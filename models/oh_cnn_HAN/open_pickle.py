import pickle



            
class IMDB_data_struct(object):

    def __init__(self,itos,stoi,label,data):
        self.itos = itos
        self.stoi = stoi
        self.label = label            
        self.data  = data


if __name__ == "__main__":
    with open('./imdb_data.pkl', 'rb') as file:

        aa = pickle.load(file)
    
    import pdb; pdb.set_trace()