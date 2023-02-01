
#some parts of this code are copied from https://github.com/NBCLab/abbr/tree/master/abbr and from

import re
import os, csv
# import spacy, nltk
import nltk
import pandas as pd



# nltk.download("punkt")


class getAbbreviations():
    def __init__(self, text):
        self.text=text


    def removeListDuplicates(self,alist):
        # create a dictionary
        myList=list(dict.fromkeys(alist))

        #dictionary to list
        return list(myList)


    def getAcronyms(self):
        text=self.text
        # acronym = re.findall('\\b[A-Z](?:[]?[A-Z]){1,}\\b', text) original

        # acronym=re.findall('(?<=\()[A-Z]+?(?=\))',text)   #round
        # acronym = re.findall('(?<=\[)[A-Z]+?(?=\])', text) #square

        # acronym=re.findall('\(([A-Za-z0-9_]+)\)',text)  # any text within brackets
        acronym = re.findall('(?<=\()[A-Za-z]+?(?=\))', text)  #modified  to include lower case and

        acronym = self.removeListDuplicates(acronym)
        return acronym


    def removePunctuation(self):
        text=self.text
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        new_words = tokenizer.tokenize(text)
        return [i.lower() for i in new_words]


    def generate_ngrams(self,s, n):
        ngrams = zip(*[s[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]


    def getFistCharacte(self,ngram_list,n):
        firstLetters=[]
        for i in range(len(ngram_list)):
            options=ngram_list[i]
            options=options.split(' ')

            for j in range(len(options)):
                firstLetters.append(options[j][0])

        result=[]
        for i in range(0,len(firstLetters),n):
            sett=firstLetters[i:i+n]
            sett=' '.join(sett)
            result.append(sett)

        remove_spaces = lambda x: re.sub(' ', '', x)
        result = list(map(remove_spaces, result))
        return result


    def getPossibleReplacemet(self):
        acronym = self.getAcronyms()
        tokens = self.removePunctuation()

        abbrev,fullName=[],[]
        for ab in acronym:
            # print('\n====================',ab)
            a=ab.lower()
            index_in_text = tokens.index(a)

            start_index=index_in_text - (len(a)+5)
            if start_index<0:
                start_index=0

            possible_expansion = tokens[start_index:index_in_text]
            nGRams = self.generate_ngrams(possible_expansion, len(a))
            firstCharacters = self.getFistCharacte(nGRams, len(a))



            for i in range(len(nGRams)):
                if firstCharacters[i] == a:
                    abbrev.append(ab)
                    fullName.append(nGRams[i])

                # else:
                #     print(editdistance.eval(a,firstCharacters[i]),ab, nGRams[i])

            if ab not in abbrev:
                abbrev.append(ab)
                fullName.append(None)

        return list(zip(abbrev,fullName))  #returns acronyms and their possible expansions

    def replaceInText(self):
        abbrevs_FullName=self.getPossibleReplacemet()
        text=self.text
        for i in abbrevs_FullName:
            try:
                text = text.replace('(' + i[0] + ')', '')
                text = text.replace(i[0], i[1])
                text=text.replace('  ',' ')
            except:
                continue

        return text  #replace acronmys with their possible full expansion


text='Background: Aromatase inhibitors (AI) have improved the prognosis, AI decrease in bone mineral density (BMD) and, bone mineral density measured via dual energy X-ray absorptiometry energy X-ray absorptiometry (DXA)'
main=getAbbreviations(text=text)
print(main.getPossibleReplacemet())
print(main.replaceInText())



