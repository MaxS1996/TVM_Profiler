import pickle
import os
import csv
import components.description_vector as dv

def dump_description_vectors(descr_vecs, metas, model_name, target, frontend, path="./"):
    if not os.path.exists(path):
        os.mkdir(path)

    folder_path = path+"/"+target
    folder_path = folder_path.replace(" ", "-")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    folder_path = path+"/"+target+"/"+model_name+"_"+frontend
    folder_path = folder_path.replace(" ", "-")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    with open(folder_path+"/"+"1.total", 'wb') as f:
        pickle.dump(descr_vecs, f)
    print("pickled %s at %s" % ("total", folder_path))

    #TODO: serialize individual 
    for name, data in metas.items():
        file_name = folder_path + "/"+name+".meta"
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        print("pickled %s at %s" % (name, file_name))

    csv_path = folder_path+".csv"
    with open(csv_path, 'w', encoding='UTF8') as csv_file:
        csv_write = csv.writer(csv_file, delimiter =";")
        csv_write.writerow(dv.DescriptionVector.csv_head())
    
        for name, data in descr_vecs.items():
            file_name = folder_path + "/"+name+".txt"
            with open(file_name, 'w') as f:
                f.write(data.to_txt())

            csv_write.writerow(data.for_csv())
            

    
        

def dump_raw_extraction(feature_vecs, name, path="./ifx_cpu"):
    folder_path = path+"_"+name

    i = 1
    test_path = folder_path
    while os.path.exists(test_path):
        test_path = folder_path + "_" + str(i)
        i += 1

    folder_path = test_path

    os.mkdir(folder_path)

    for name, data in feature_vecs.items():
        file_name = folder_path + "/"+name+".pckl"
        with open(file_name, 'wb') as f:
            pickle.dump(data[0], f)
        print("pickled %s at %s" % (name, file_name))

        with open(folder_path+"/"+name+".txt", 'w') as f:
            head = []
            head += data[1].descriptions

            if head[-1] != "execution time":
                head.append("execution time")

            values = data[1].get_collected_output()
            
            if len(data) >= 3:
                values.append(data[2])
            else:
                values.append("not in final schedule")

            write = csv.writer(f, delimiter =";") 
            for i in range(0, len(head)):
                fields = [head[i], values[i]]
                write.writerow(fields)
    
    print("pickled everything")
    return
