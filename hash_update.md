### Hash Update

| File 1 = `my_file.py` |hash| File 2  = `my_file.py` |hash|ChangeNd| 
|----------|---|----------|--|----| 
| Class 1    |fsadfwe| Class 1   | fsadfwe|N| 
| Class 2    |berre| Class 2   | berre|N| 
| Class 3    |brenjrwgr| Class 3   |brenjrwgr |N| 
| function 1    |bsgreghwe5| function 1   |bsgreghwe5 |N| 
| function 2    |b345rhw5hwer |function 2   |b345rhw5hwer |N| 
| function 3    |5hjwjntrtn|function 3   |5hjwjntr |Y| 

### Hash Initialisation

- Compute the hash of file.
 - Compute the Hash of all classes and Store it in A dictionary {class_name: hash} 
 - Compute the Hash of all functions and Store it in A dictionary {function_name: hash}

### Hash Comparison

cfobj = Classes/functions object

Repeat Hash Initialisation for the new file if the hash of the file has changed.

**Case** **1**: Hash of the file has changed and Number of classes and functions are same. 

**Possibilities**: 
 - Any one of the Class/function has been changed.
 - Any one of the Class/function has been added by deleting another class/function.

**Solution**:
 1. Separate all the cfobj whose hashes are still in both the files and the cfobj whose hashes are not in both the files `(cfobj_unk)`.
 2. Delete all the cfobj from the database  which is not in the new file. 
 3. Embed the new cfobj and store in the database with a foreign key of the file.

**Case** **2**: Hash of the file has changed and Number of classes and functions are not same.

**Possibilities**:
 - cfobj have been added or deleted.

**Solution**:
 - Repeat Steps 1,2,3 from the case 1.


Conclusion
---

Begin by checking whether all the old hashes are there in the new version of the file. Delete all the entries from the database that is not in the new file.

Then embed the new cfobj in the database.   

