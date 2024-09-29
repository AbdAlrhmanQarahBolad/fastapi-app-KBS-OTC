from fastapi import FastAPI, status
from transformers import AutoTokenizer
import stanza
from experta import *
import re
from fastapi.responses import JSONResponse
from typing import Dict, Any

app = FastAPI()

# Declare global variable for the Stanza pipeline
nlp = None
#engine = None
mp = {}
# Download Stanza Arabic model and set up pipeline on startup
@app.on_event("startup")
async def load_nlp_model():
    global nlp
    #global engine
    #stanza.download('ar')  # Downloads the Arabic model
    nlp = stanza.Pipeline('ar',download_method=None)  # Sets up the Stanza pipeline
    
    


@app.get("/data/")
def root():
    return {"data":mp}


@app.post("/{sen}")
def runEngine(sen: str,data: Dict[Any, Any]):
    
    engine = DialogManager() 

    arr=pipline(sen)
    for k, v in arr.items():
        data.update({k:v})
    engine.declareFacts(data)
    engine.run()
    if (engine.sentence ==""):
        engine.sentence ="خرجنا من نطاق الادوية المتاحة من دون وصفة , يمكنك زيارة طبيب "
        engine.endFlag = True
        
    return JSONResponse(status_code=status.HTTP_200_OK, content={"Message":engine.sentence,"EndFlag":engine.endFlag,'data':data})

@app.post("/reset/")
def resetEngine():
    #global mp 
    #mp ={}
    #engine.reset()
    #engine.run()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"Message":"The engine has been reset successfully"})





def pipline(sentence):
    
    sentence = convert_to_lemma_sentence_without_diacritics(sentence)
    #print(sentence)
    doc = nlp(sentence)
    # Identify conjunctions and split the sentence
    split_sentences = []
    temp_sentence = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos != 'CCONJ':
                temp_sentence.append(word.text)
            if word.upos == 'CCONJ':  # Check if it's a conjunction
                split_sentences.append(' '.join(temp_sentence).strip())
                temp_sentence = []  # Start a new sentence
        if temp_sentence:
            split_sentences.append(' '.join(temp_sentence).strip())
    sentences =[]
    my_dict = {}
    #sentence = "لا أعاني من الإسهال و الإسهال المائي و الإسهال الغير مائي و الإسهال ذو قوام دهني و ألم بطن و رائحة كريهة و غثيان دون إقياء و دون تشنج وامساك و غثيان و إقياء حتى السوائل و المياه و حرقة في المريء و عسر هضم و تجشؤ و صداع قفوي و صداع جبهي و صداع نصفي و سعال ديكي و سعال رطب و سعال مع قشع و ضيق تنفس و سعال جاف و سيلان و احتقان ليلي و احتقان أنفي و زكام و سيلان الأنف و الرشح التحسسي و العطاس"
    lemmas = ['إسهال','زاكم','سعال',
            'إسهال غير مائي','إسهال مائي','إسهال دهني','ألم بطن','رائحة كريه','رائحة',
            'غثيان','إقياء شديد','تشنج','حرارة',
            'إمساك','إقياء',
            'حرقة','حرقة مريء','عسر هضم',
            'تجشؤ','صداع قفوي','صداع جبهي',
            'صداع','صداع نصفي','سعال ديكي','قيئ',
            'سعال رطب','ضيق تنفس','الإسهال غير مائي','الإسهال مائي','الإسهال دهني',
            'سعال جاف','سيلان','احتقان','احتقان ليلي','احتقان أنفي','سيلان أنف','رشح','عطاس','العطاس',
            'سعال الديكي','قيء','قيء شديد','السيلان','احتقان انفي','احتقان أنفى','احتقان أنف','احتقان الانفي']
    lemmas_3word = ['إسهال غير مائي','الإسهال غير مائي']
    lemmas_2word = ['احتقان الانفي','احتقان أنف','احتقان أنفى','قيء شديد','احتقان انفي','سعال الديكي','الإسهال مائي','الإسهال دهني','إسهال مائي','إسهال دهني','ألم بطن','رائحة كريه','إقياء شديد','حرقة مريء','احتقان ليلي','احتقان أنفي','سيلان أنف','عسر هضم','صداع قفوي','سعال جاف','صداع جبهي','صداع نصفي','سعال ديكي','سعال رطب','ضيق تنفس',]
    for sentence in split_sentences:
        #####lemma_sentence = convert_to_lemma_sentence_without_diacritics(sentence)
        lemma_sentence = sentence
        if lemma_sentence in lemmas :
            my_dict[lemma_sentence] = 2
            continue
        negated, details = is_sentence_negated(sentence)
        split_sentence = sentence.split(" ")
        three_word =[]
        for i in range(len(split_sentence)-2):
            three_word.append(split_sentence[i]+" "+split_sentence[i+1]+" "+split_sentence[i+2])
        
        two_word =[]
        for i in range(len(split_sentence)-1):
            two_word.append(split_sentence[i]+" "+split_sentence[i+1])
        
        for trios in three_word :
            ####lemma_sentence = convert_to_lemma_sentence_without_diacritics(trios)
            lemma_sentence = trios
            if lemma_sentence in lemmas_3word :
                if negated:
                    my_dict[lemma_sentence] = 0
                else :
                    my_dict[lemma_sentence] = 1
                break
        
        for dous in two_word :
            ####lemma_sentence = convert_to_lemma_sentence_without_diacritics(dous)
            lemma_sentence=dous
            if lemma_sentence in lemmas_2word :
                if negated:
                    my_dict[lemma_sentence] = 0
                else :
                    my_dict[lemma_sentence] = 1
                break
        for word in split_sentence :
            ####lemma_word=get_lemma_without_diacritics(word)
            lemma_word=word
            if lemma_word in lemmas :
                if negated:
                    my_dict[lemma_word] = 0
                else :
                    my_dict[lemma_word] = 1
                break
    for i in range(len(list(my_dict.items()))):
        if (list(my_dict.items())[i][1]==2):
            my_dict[list(my_dict.items())[i][0]] = list(my_dict.items())[i-1][1]
    return my_dict

def is_sentence_negated(sentence):
    negation_words = {'لا', 'لم', 'ليس', 'ما', 'ما زال'}
    doc = nlp(sentence)
    negation_present = False
    negation_details = []

    for sent in doc.sentences:
        for word in sent.words:
            if word.text in negation_words:
                negation_present = True
                head_word = sent.words[word.head - 1] if word.head > 0 else None
                negation_details.append({
                    'negation_word': word.text,
                    'negated_word': head_word.text if head_word else None,
                    'relation': word.deprel
                })

    return negation_present, negation_details

def remove_diacritics(text):
    # Arabic diacritics Unicode range: U+064B to U+0652
    return re.sub(r'[\u064B-\u0652]', '', text)

# Function to get lemma of a single word without diacritics
def get_lemma_without_diacritics(word):
    # Process the word using Stanza
    doc = nlp(word)

    # Extract the lemma of the first token
    lemma = doc.sentences[0].words[0].lemma


    # Remove diacritics from the lemma
    lemma_without_diacritics = remove_diacritics(lemma)

    return lemma_without_diacritics

def convert_to_lemma_sentence_without_diacritics(sentence):
    # Process the sentence using Stanza
    doc = nlp(sentence)

    # Extract the lemma for each word and remove diacritics
    lemma_sentence = " ".join([remove_diacritics(word.lemma) for sentence in doc.sentences for word in sentence.words])


    return lemma_sentence

def split_by_every_third_space(string):
    # Split the string by spaces
    words = string.split(' ')

    # Initialize variables
    substrings = []
    current_group = []

    # Iterate through words and group them
    for i, word in enumerate(words):
        current_group.append(word)
        # Every second space (i.e., every two words) create a new substring
        if (i + 1) % 3 == 0:
            substrings.append(' '.join(current_group))
            current_group = []

    # If there are remaining words that didn't form a complete group
    if current_group:
        substrings.append(' '.join(current_group))

    return substrings

def split_by_every_second_space(string):
    # Split the string by spaces
    words = string.split(' ')

    # Initialize variables
    substrings = []
    current_group = []

    # Iterate through words and group them
    for i, word in enumerate(words):
        current_group.append(word)
        # Every second space (i.e., every two words) create a new substring
        if (i + 1) % 2 == 0:
            substrings.append(' '.join(current_group))
            current_group = []

    # If there are remaining words that didn't form a complete group
    if current_group:
        substrings.append(' '.join(current_group))

    return substrings

def create_instance(class_name,param):
    if class_name in globals():
        return globals()[class_name](param)
    elif class_name in locals():
        return locals()[class_name](param)
    else:
        raise ValueError(f"Class '{class_name}' is not defined.")








# Define the facts
class رشح(Fact):
    pass
class سيلان(Fact):
    pass
class السيلان(Fact):
    pass
class سيلان_أنف(Fact):
    pass
class إسهال(Fact):
    pass
class إسهال_مائي(Fact):
    pass
class إسهال_غير_مائي(Fact):
    pass
class إسهال_دهني(Fact):
    pass
class الإسهال_مائي(Fact):
    pass
class الإسهال_غير_مائي(Fact):
    pass
class الإسهال_دهني(Fact):
    pass
class ألم(Fact):
    pass
class ألم_بطن(Fact):
    pass
class رائحة(Fact):
    pass
class رائحة_كريه(Fact):
    pass
class حرارة(Fact):
    pass
class إمساك(Fact):
    pass
class حرقة(Fact):
    pass
class حرقة_مريء(Fact):
    pass
class عسر_هضم(Fact):
    pass
class تجشؤ(Fact):
    pass
class إقياء(Fact):
    pass
class قيئ(Fact):
    pass
class قيء(Fact):
    pass
class قيء_شديد(Fact):
    pass
class إقياء_شديد(Fact):
    pass
class غثيان(Fact):
    pass
class تشنج(Fact):
    pass
class عطاس(Fact):
    pass
class العطاس(Fact):
    pass
class احتقان(Fact):
    pass
class احتقان_أنفي(Fact):
    pass
class احتقان_أنف(Fact):
    pass
class احتقان_انفي(Fact):
    pass
class احتقان_أنفى(Fact):
    pass
class احتقان_الانفي(Fact):
    pass
class احتقان_ليلي(Fact):
    pass
class زاكم(Fact):
    pass
class سعال(Fact):
    pass
class سعال_ديكي(Fact):
    pass
class سعال_الديكي(Fact):
    pass
class سعال_رطب(Fact):
    pass
class سعال_جاف(Fact):
    pass
class ضيق_تنفس(Fact):
    pass
class صداع(Fact):
    pass
class صداع_قفوي(Fact):
    pass
class صداع_جبهي(Fact):
    pass
class صداع_نصفي(Fact):
    pass
class Intent(Fact):
    pass
class empty(Fact):
    pass


# Define the Knowledge Base
class DialogManager(KnowledgeEngine):
    sentence = ""
    endFlag = False
    @DefFacts()
    def initial_facts(self):
        yield empty(True)

    def declareFacts(self,exist):
        for elem in exist:
            className=elem.replace(" ", "_")
            self.declare(create_instance(className,exist[elem]))
        #if not bool(exist) :
        #   self.declare(empty(True))
###################

    @Rule(إسهال(True),
        AND(NOT(إسهال_مائي()),NOT(الإسهال_مائي())),
        AND(NOT(إسهال_دهني()),NOT(الإسهال_دهني())),
        AND(NOT(إسهال_غير_مائي()),NOT(الإسهال_غير_مائي())),
        )
    def ask_DiarrheaState1(self):
        self.sentence = "ما هو قوام الاسهال (مائي , غير مائي ,دهني) الذي تعاني منه ؟ , اجب اجابة كاملة"
        self.endFlag = False
        return 
        #user_response = input("ما هو قوام الاسهال (مائي , غير مائي ,دهني) الذي تعاني منه ؟ , اجب اجابة كاملة ")
        #self.arr=pipline(user_response)
        #declareFacts(self.arr)

    @Rule(OR(إسهال_غير_مائي(True),الإسهال_غير_مائي(True)) , NOT(ألم_بطن()) )
    def ask_DiarrheaState2(self):
        self.sentence ="هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        #user_response = input("هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة ")
        #self.arr=pipline(user_response)
        #declareFacts(self.arr)

    @Rule(OR(إسهال_غير_مائي(True),الإسهال_غير_مائي(True)) , NOT(رائحة_كريه()) )
    def ask_DiarrheaState3(self):
        self.sentence ="هل تعاني من  رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        #user_response = input("هل تعاني من  رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة ")
        #self.arr=pipline(user_response)
        #declareFacts(self.arr)

    @Rule(  OR(إسهال_مائي(True),الإسهال_مائي(True)), NOT(ألم_بطن()))
    def Diarrheamedicen55(self):
        self.sentence ="هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule( OR(إسهال_مائي(True),الإسهال_مائي(True)), NOT(رائحة_كريه()))
    def Diarrheamedicen55(self):
        self.sentence ="هل تعاني من رائحة كريه مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من رائحة كريه مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(  OR(إسهال_مائي(True),الإسهال_مائي(True)), رائحة_كريه(False),NOT(ألم_بطن()))
    def Diarrheamedicen56(self):
        self.sentence ="هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(  OR(إسهال_مائي(True),الإسهال_مائي(True)), NOT(رائحة_كريه()),ألم_بطن(False))
    def Diarrheamedicen57(self):
        self.sentence ="هل تعاني من رائحة كريه مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من رائحة كريه مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(  OR(إسهال_مائي(True),الإسهال_مائي(True)), ألم_بطن(False),رائحة_كريه(False),salience=1000)
    def Diarrheamedicen1(self):
        self.sentence ="الدواء المناسب لحالتك هو ايديوم او دياريديوم"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو ايديوم او دياريديوم")
        self.halt()

    @Rule(  OR(إسهال_غير_مائي(True),الإسهال_غير_مائي(True))
    , OR(AND(ألم_بطن(True),رائحة_كريه(False)),
     AND(ألم_بطن(False),رائحة_كريه(True)),
       AND(ألم_بطن(True),رائحة_كريه(True)))
    ,NOT(حرارة()))
    def ask_DiarrheaState4(self):
        self.sentence ="هل تعاني من  حرارة مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        #user_response = input("هل تعاني من  حرارة مع الاسهال ؟ , اجب اجابة كاملة ")
        #self.arr=pipline(user_response)
        #declareFacts(self.arr)

    @Rule( OR(إسهال_غير_مائي(True),الإسهال_غير_مائي(True))
    , OR(ألم_بطن(True),رائحة_كريه(True)),حرارة(False),salience=1000)
    def Diarrheamedicen2(self):
        self.sentence = "الدواء المناسب لحالتك هو ديافوريل او ميديوفوريل او اينترفوريل"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو ديافوريل او ميديوفوريل او اينترفوريل")
        self.halt()

    @Rule(OR(إسهال_غير_مائي(True),الإسهال_غير_مائي(True))
    , OR(AND(ألم_بطن(True),رائحة_كريه(False)),
     AND(ألم_بطن(False),رائحة_كريه(True)),
       AND(ألم_بطن(True),رائحة_كريه(True))),حرارة(True),salience=1000)
    def Diarrheamedicen3(self):
        self.sentence = "الدواء المناسب لحالتك هو باترام او سيبترين"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو باترام او سيبترين")
        self.halt()


    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), ألم_بطن(True),رائحة_كريه(True) ,salience=1000)
    def Diarrheamedicen4(self):
        self.sentence = "الدواء المناسب لحالتك هو فلاجيل او ميترونيدازول"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو فلاجيل او ميترونيدازول")
        self.halt()
######################alawi
    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), NOT(ألم_بطن()),NOT(رائحة_كريه()) )
    def Diarrheamedicen5(self):
        self.sentence = "هل تعاني من ألم بطن او رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ألم بطن او رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), NOT(رائحة_كريه()) )
    def Diarrheamedicen6(self):
        self.sentence = "هل تعاني من رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), NOT(ألم_بطن()) )
    def Diarrheamedicen7(self):
        self.sentence = "هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ألم بطن مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), ألم_بطن(True),NOT(رائحة_كريه()) )
    def Diarrheamedicen8(self):
        self.sentence = "هل تعاني من رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من رائحة كريهة مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(الإسهال_دهني(True),إسهال_دهني(True) ), رائحة_كريه(True),NOT(ألم_بطن()) )
    def Diarrheamedicen9(self):
        self.sentence = "هل تعاني من ألم_بطن مع الاسهال ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ألم_بطن مع الاسهال ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
#امساك
    @Rule(إمساك(True),salience=1000 )
    def holdingmedicen1(self):
        self.sentence = " الادوية المناسبه لحالتك هي تحاميل غليسرين او شراب لاكتوز او دوفلاك زيت خروع او حب سنامكي او سينوزيد (لا يزيد اسختدامها عن 5 ايام ) او حب لاكسين او ديلاكسا (لا يزيد استخدامها عن 5 ايام ) او حقنة شرجية  "
        self.endFlag = True
        return
        print(":الادوية المناسبه لحالتك هي")
        print("تحاميل غليسرين")
        print("شراب لاكتوز او دوفلاك")
        print("زيت خروع")
        print("حب سنامكي او سينوزيد (ملاحظة : لايزيد استخدامها عن 5 ايام)")
        print("حب لاكسين او ديلاكسا (ملاحظة : لايزيد استخدامها عن 5 ايام)")
        print("حقنة شرجية")
        self.halt()

#اقياء وغثيان
    @Rule(OR(إقياء(True),قيء(True),قيئ(True)),NOT(إقياء_شديد()),NOT(قيء_شديد()))
    def ask_neaseaState01(self):
        self.sentence = "هل تعاني من  اقياء شديد ام لا ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return


    @Rule(غثيان(True), NOT(إقياء()),NOT(قيء()),NOT(قيئ()),NOT(إقياء_شديد()),NOT(قيء_شديد()))
    def ask_neaseaState1(self):
        self.sentence = "هل تعاني من  (اقياء ام اقياء شديد ام لا تعاني ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من  (اقياء ام اقياء شديد ام لا تعاني ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(OR(إقياء(True),قيء(True),قيئ(True)), NOT(غثيان()))
    def ask_neaseaState2(self):
        self.sentence = "هل تعاني من غثيان ؟ , اجب اجابة كاملة"
        self.endFlag = False
        return
        user_response = input("هل تعاني من غثيان مع الاقياء ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(OR(OR(إقياء(True),قيء(True),قيئ(True)), غثيان(True)), NOT(ألم_بطن()))
    def ask_neaseaState3(self):
        self.sentence = "هل تعاني من  ألم البطن ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من  ألم البطن ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(OR(OR(إقياء(True),قيء(True),قيئ(True)), غثيان(True)), NOT(تشنج()))
    def ask_neaseaState4(self):
        self.sentence = "هل تعاني من  تشنج ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من  تشنج ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(غثيان(True),OR(إقياء(False),قيء(False),قيئ(False)),تشنج(False) ,salience=1000)
    def neaseamedicen1(self):
        self.sentence = "الادوية المناسبة لحالتك هي دي فوميت او اونفران"
        self.endFlag = True
        return
        print("الادوية المناسبة لحالتك هي دي فوميت او اونفران")
        self.halt()

    @Rule(AND(OR(OR(إقياء(True),قيء(True),قيئ(True)), غثيان(True))), ألم_بطن(True), تشنج(True),salience=1000 )
    def neaseamedicen2(self):
        self.sentence = "الادوية المناسبه لحالتك هي موتيلوسير او موتين"
        self.endFlag = True
        return
        print("الادوية المناسبه لحالتك هي موتيلوسير او موتين")
        self.halt()



    @Rule(OR(قيء_شديد(True),إقياء_شديد(True)), غثيان(True),salience=1000)
    def neaseamedicen3(self):
        self.sentence = "الادوية المناسبه لحالتك هي حب مص مثل كاميترون او فومي سيت او حقن مثل فومي كايند او اندانسترون"
        self.endFlag = True
        return
        print(":الادوية المناسبه لحالتك هي")
        print(" حب مص : كاميترون او فومي سيت")
        print("حقن : فومي كايند او اندانسترون")
        self.halt()

#حرقة
    @Rule(حرقة(True),salience=1000 )
    def Heartburnmedicen1(self):
        self.sentence = "الادوية المناسبه لحالتك هي حب مص مثل ريني فيرس او رينوفول او حب بلع مثل ايزوستوم او لانسوبرازول او بنتا او اذا كنت تتناول كلوبيد غريل فالدواء هو ديسكا لانسوبرازول او بنتا او بانتوبرال "
        self.endFlag = True
        return
        print(":الادوية المناسبه لحالتك هي")
        print("حب مص : ريني فيرس او رينوفول")
        print(" حب بلع : ايزوستوم او لانسبورازول او بنتا")
        print("اذا كنت تتناول كلوبيد غريل غريل فالدواء هو : ديسكا لانسوبرازول او بنتا او بانتوبرال")
        self.halt()

#عسر هضم تجشؤ
    @Rule(عسر_هضم(True),NOT(تجشؤ()) )
    def ask_IndigestionState1(self):
        self.sentence ="هل تعاني من  تجشؤ  مع عسر الهضم؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من  تجشؤ  مع عسر الهضم؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(تجشؤ(True),NOT(عسر_هضم()) )
    def ask_IndigestionState2(self):
        self.sentence ="هل تعاني من  عسر الهضم مع التجشؤ ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من  عسر الهضم مع التجشؤ ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)
    @Rule(تجشؤ(True) , عسر_هضم(True),salience=1000)
    def Indigestionmedicen1(self):
        self.sentence =" الادوية المناسبة لحالتك هي: موتالون او سباسموكولوناز او ميزيم فورت"
        self.endFlag = True
        return
        print(" الادوية المناسبة لحالتك هي: موتالون او سباسموكولوناز او ميزيم فورت")
        self.halt()


####################
    @Rule(OR(سيلان(True),السيلان(True)),NOT(رشح()),AND(NOT(عطاس()),NOT(العطاس())))
    def ask_s6(self):
        self.sentence ="هل تعاني من الرشح او العطاس  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من الرشح او العطاس  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(رشح(True),AND(NOT(السيلان()),NOT(سيلان())),AND(NOT(عطاس()),NOT(العطاس())))
    def ask_s7(self):
        self.sentence ="هل تعاني من سيلان او العطاس  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من سيلان او العطاس  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(عطاس(True),العطاس(True)),AND(NOT(السيلان()),NOT(سيلان())),NOT(رشح()))
    def ask_s8(self):
        self.sentence ="هل تعاني من سيلان او رشح  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من سيلان او رشح  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(سيلان(True),السيلان(True)),رشح(True),AND(NOT(عطاس()),NOT(العطاس())))
    def ask_s9(self):
        self.sentence ="هل تعاني من العطاس  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من العطاس  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(رشح(True),NOT(سيلان()),NOT(السيلان()),OR(عطاس(True),العطاس(True)))
    def ask_s10(self):
        self.sentence ="هل تعاني من سيلان   ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من سيلان   ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(عطاس(True),العطاس(True)),OR(سيلان(True),السيلان(True)),NOT(رشح()))
    def ask_s11(self):
        self.sentence ="هل تعاني من   رشح  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من   رشح  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)


    @Rule(احتقان(True),NOT(احتقان_ليلي()),NOT(احتقان_أنفي()),NOT(احتقان_انفي()),NOT(احتقان_أنفى()),NOT(احتقان_أنف()),NOT(احتقان_الانفي()))
    def ask_s3(self):
        self.sentence ="هل تعاني من احتقان انفي ام احتقان ليلي  ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من احتقان انفي ام احتقان ليلي  ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)


    @Rule(OR(احتقان_أنفي(True),احتقان_انفي(True),احتقان_أنفى(True),احتقان_أنف(True),احتقان_الانفي(True)),NOT(زاكم()))
    def ask_s2(self):
        self.sentence ="هل تعاني من الزكام ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من الزكام ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(زاكم(True),NOT(احتقان_أنفي()),NOT(احتقان_انفي()),NOT(احتقان_أنفى()),NOT(احتقان_أنف()),NOT(احتقان_الانفي()))
    def ask_s4(self):
        self.sentence ="هل تعاني من ٱحتقان الأنفي ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ٱحتقان الأنفي ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(سيلان(True),السيلان(True)),NOT(احتقان_ليلي()))
    def ask_s5(self):
        self.sentence ="هل تعاني من احتقان ليلي ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من احتقان ليلي ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(احتقان_ليلي(True),NOT(سيلان()),NOT(السيلان()))
    def ask_s1(self):
        self.sentence ="هل تعاني من سيلان الانف ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من سيلان الانف ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)

        declareFacts(self.arr)

    @Rule(سعال(True),NOT(سعال_جاف()),NOT(سعال_رطب()),NOT(سعال_ديكي()),NOT(سعال_الديكي()))
    def ask_kindOfCough(self):
        self.sentence ="هل تعاني من السعال الجاف او السعال الرطب او السعال الديكي؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من السعال الجاف او السعال الرطب او السعال الديكي؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)

        declareFacts(self.arr)


    @Rule(سعال(True),سعال_رطب(True))
    def ask_ShortnessOfBreath(self):
        self.sentence ="هل تعاني من ضيق التنفس ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من ضيق التنفس ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)

        declareFacts(self.arr)

    @Rule(صداع(True),NOT(صداع_قفوي()),NOT(صداع_جبهي()),NOT(صداع_نصفي()))
    def ask_kindOfHeadache(self):
        self.sentence ="هل تعاني من الصداع النصفي او الصداع الجبهي او الصداع القفوي ؟ , اجب اجابة كاملة "
        self.endFlag = False
        return
        user_response = input("هل تعاني من الصداع النصفي او الصداع الجبهي او الصداع القفوي ؟ , اجب اجابة كاملة ")
        self.arr=pipline(user_response)
        declareFacts(self.arr)

    @Rule(OR(عطاس(True),العطاس(True)),
          OR(سيلان(True),السيلان(True))
          ,رشح(True),salience=1000)
    def m1(self):
        self.sentence ="الدواء المناسب لحالتك هو كلاريتين او زيتريزين او ليفوندا"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو كلاريتين او زيتريزين او ليفوندا")
        self.halt()

    @Rule( OR(احتقان_أنفي(True),احتقان_انفي(True),احتقان_أنفى(True),احتقان_أنف(True),احتقان_الانفي(True))
    ,
          زاكم(True),salience=1000)
    def m2(self):
        self.sentence ="الدواء المناسب لحالتك هو يونادول(كولد اند فلو) او بارادرين (كولد اند فلو ) او رينومود او كريب ستوب"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو يونادول(كولد اند فلو) او بارادرين (كولد اند فلو ) او رينومود او كريب ستوب")
        self.halt()

    @Rule( OR(سيلان(True),السيلان(True))
    ,
        احتقان_ليلي(True),salience=1000)
    def m3(self):
        self.sentence ="الدواء المناسب لحالتك هو تولين اكسترا او رينومود او فلوريكس"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو تولين اكسترا او رينومود او فلوريكس")
        self.halt()

    @Rule(سعال_جاف(True),salience=1000)
    def m4(self):
        self.sentence ="الدواء المناسب لحالتك هو هوستاجيل او ثايمكس او توبلكسيل او غواميزيم او كوفيستا"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو هوستاجيل او ثايمكس او توبلكسيل او غواميزيم او كوفيستا")
        self.halt()

    @Rule(AND(سعال_رطب(True),ضيق_تنفس(False)),salience=1000)
    def m5(self):
        self.sentence ="الدواء المناسب لحالتك هو ازمكس او ازماديكس او بيلكافون او موكولار او بروسبان او ريسيبان او ثيموجل بلس"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو ازمكس او ازماديكس او بيلكافون او موكولار او بروسبان او ريسيبان او ثيموجل بلس")
        self.halt()

    @Rule(AND(سعال_رطب(True),ضيق_تنفس(True)),salience=1000)
    def m6(self):
        self.sentence ="الدواء المناسب لحالتك هو بلموفيلليين او بنتو فلليين او اوفلليين او نيوفللين"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو بلموفيلليين او بنتو فلليين او اوفلليين او نيوفللين")
        self.halt()


    @Rule(OR(سعال_ديكي(True),سعال_الديكي(True)),salience=1000)
    def m7(self):
        self.sentence ="الدواء المناسب لحالتك هو سالبوتامول"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو سالبوتامول")
        self.halt()

    @Rule(صداع_قفوي(True),salience=1000)
    def m8(self):
        self.sentence ="الدواء المناسب لحالتك هو باراسيتامول او برودول جوينت او يرجى قياس الضغط"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو باراسيتامول او برودول جوينت او يرجى قياس الضغط")
        self.halt()

    @Rule(صداع_جبهي(True),salience=1000)
    def m9(self):
        self.sentence ="الدواء المناسب لحالتك هو برودول بلس ك او اجيلوموكس "
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو برودول بلس ك او اجيلوموكس ")
        self.halt()

    @Rule(صداع_نصفي(True),salience=1000)
    def m10(self):
        self.sentence ="الدواء المناسب لحالتك هو يوني اكسدرين او بارادرين شقيقة"
        self.endFlag = True
        return
        print("الدواء المناسب لحالتك هو يوني اكسدرين او بارادرين شقيقة")
        self.halt()



