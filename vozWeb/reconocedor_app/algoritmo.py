# -*- encoding: utf-8 -*-
###############################################################################
## APROXIMACIÓN POR REDES NEURONALES RECURSIVAS
###############################################################################

###############################################################################
## Globales
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
###############################################################################

###############################################################################
## Librerías
import math
import os
import random
import sys
import time
import logging
import gzip
import re
import tarfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from six.moves import urllib
from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
###############################################################################

###############################################################################
## Vocabulario especial
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
###############################################################################

###############################################################################
## Expresiones regulares para partir la frase
_WORD_SPLIT = re.compile(b"([.,?\"';)(])")
_DIGIT_RE = re.compile(br"\d")
###############################################################################

###############################################################################
## Parametrización
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS
###############################################################################

###############################################################################
## Buckets de pregunta - respuesta
_buckets = [(40, 10)]
###############################################################################

###############################################################################
## Carga de datos
def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set
###############################################################################

###############################################################################
## Parseador
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]
###############################################################################
  
###############################################################################
## Construcción del universo de palabras
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not tf.gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with tf.gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
###############################################################################

###############################################################################
## Inicialización del universo de palabras
def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if tf.gfile.Exists(vocabulary_path):
    rev_vocab = []
    with tf.gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)
###############################################################################

###############################################################################
## Conversión de una frase a enteros para el modelo
def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
###############################################################################

###############################################################################
## Conversión de los datos de entrenamiento para el modelo
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not tf.gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with tf.gfile.GFile(data_path, mode="rb") as data_file:
      with tf.gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
###############################################################################
            
###############################################################################
## Creación del modelo
def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size,
      FLAGS.vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt :
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
  return model
  
  
  
###############################################################################
## Conversión salario experiencia 1
## 	esta función transforma 
##	ofertas de más de 28500 € con experiencia de 18 meses
##	en
##  ofertas de más de <sal:28500 euros> con experiencia de <exp: Al menos 1 año>


def conversion_salario_experiencia_1(sentence):
    # coding=utf8

    import re

    # Expresiones regulares
    sueldo = re.compile(r"\s\d{0,3}[.,]?(\d{3})((\s*euros)|(\s*eur)|(\s*€))?(.*?(\bmes\b|\baño\b))?",re.UNICODE)
    experiencia = re.compile(r"(más de|mas de|al menos)?(no necesitar|no requerir|sin experiencia|sin nada de experiencia|(\s(\d{1,2}|uno|un|dos|tres|cuatro|cinco)\s(meses|mes|años|año)?))"
                                                                                                     ,re.UNICODE)
    numero_sueldo = re.compile(r"\s\d{0,3}[.,]?(\d{3})",re.UNICODE)
    numero_experiencia = re.compile(r"(\s\d{1,2}\s|(uno|un|dos|tres|cuatro|cinco|no necesitar|no requerir|sin experiencia|sin nada de experiencia))",re.UNICODE)

    mes = re.compile(r"mes|meses",re.UNICODE)


    # Funcion para eliminar palabras sobrantes
    def limpieza(line):
            line = re.sub(r'euros |eur |por año|al año|cada año|al mes|por mes|cada mes|\.|,|\?|\¿|\!|\¡|\*','',line)
            return line

    # Lexematizacion
    def lexematizar(line):
        line = re.sub(r'no necesit\w*','no necesitar',line)
        line = re.sub(r'no requ\w*','no requerir',line)
        return line

    def convertir_texto(line):        
  
        line = line.lower()
        line = re.sub('€','euros',line)
        line = re.sub(' +',' ',line)

        # Lexematización
        line = lexematizar(line)              



        # Extracción de SUELDO        

        result1 = sueldo.search(line)

        # Utilizo esta variable para los sueldos que están dados como rango
        es_mensual = False

        while (result1 is not None):
            numero_texto = numero_sueldo.search(result1.group())
            numero = re.sub('\.','',numero_texto.group())
            numero = re.sub(' +','',numero)
            numero = int(re.sub(',','',numero))

            #Si está en meses lo pasamos a año
            if es_mensual or mes.search(result1.group()):
                numero = numero * 12
                es_mensual = True 

            #Reemplazamos 
            reemplazo = ' <sal:'+ str(numero) + ' euros> '
            line = re.sub(numero_texto.group(),reemplazo,line)

            # Comprobamos si se puede reemplazar otro sueldo
            result1 = numero_sueldo.search(line)



        # Extracción de EXPERIENCIA
        result2 = experiencia.search(line)

        if(result2 is not None):
            numero_exp = numero_experiencia.search(result2.group())
            numero_exp = re.sub('un|uno','1',numero_exp.group())
            numero_exp = re.sub('dos','2',numero_exp)
            numero_exp = re.sub('tres','3',numero_exp)
            numero_exp = re.sub('cuatro','4',numero_exp)
            numero_exp = re.sub('cinco','5',numero_exp)
            numero_exp = re.sub(r'sin experiencia|sin nada de experiencia|no necesitar|no requerir','0',numero_exp)

            numero_exp = float(numero_exp)

            #Si está en meses lo pasamos a años
            if mes.search(result2.group()):
                numero_exp = numero_exp / 12.

            if numero_exp >= 10:
                cadena = ' <exp: Más de 10 años> '
            elif numero_exp >= 5:
                cadena = ' <exp: Más de 5 años> '
            elif numero_exp >= 4:
                cadena = ' <exp: Al menos 4 años> '
            elif numero_exp >= 3:
                cadena = ' <exp: Al menos 3 años> ' 
            elif numero_exp >= 2:
                cadena = ' <exp: Al menos 2 años> '
            elif numero_exp >= 1:
                cadena = ' <exp: Al menos 1 año> '
            else:
                cadena = ' <exp: No Requerida> '

            # Reemplazamos 
            line = re.sub(result2.group(),cadena,line)

        line = limpieza(line)
        #print(type(line))
        return line
    
    # LLamada a la función
    texto2 = convertir_texto(sentence)

    return(texto2)

###############################################################################


###############################################################################
## Conversión salario experiencia 2
## 	esta función transforma 
##	ofertas de más de <sal:28500 euros> con experiencia de <exp: Al menos 1 año>
##	en
##  ofertas de más de <sal:Salario1> con experiencia de <experiencia1>
## [28500]
##	[Al menos un año]

def conversion_salario_experiencia_2(sentence):

    import re

    def preparacion(line):
        # Definimos las expresiones regulares para encontrar las etiquetas de sueldo y experiencia
        regex_sueldo = re.compile(r"<sal:\d.*?>",re.UNICODE)
        regex_experiencia = re.compile(r"<exp:.*?>",re.UNICODE)

        # Expresiones para obtener el contenido de las etiquetas
        regex_valor_sueldo = re.compile(r"\d{3,}",re.UNICODE)
        regex_valor_exp = re.compile(r": .*?>",re.UNICODE) # Como primero reemplazaremos sueldo no habrá problema con esta regex

        # SALARIO
        etiq_sueldo = regex_sueldo.search(line)

        iteracion = 1
        valor_sueldo = []
        valor_exp = []
        while(etiq_sueldo is not None):
            reemplazo = '<salario' + str(iteracion) + '>'
            line = re.sub(etiq_sueldo.group(),reemplazo,line)

            # Obtengo el valor numérico del sueldo para guardarlo en un array
            sueldo = regex_valor_sueldo.search(etiq_sueldo.group())
            valor_sueldo.append(int(sueldo.group()))

            # Comprobamos si se puede reemplazar otro sueldo
            etiq_sueldo = regex_sueldo.search(line)
            iteracion = iteracion + 1


        # EXPERIENCIA
        etiq_exp = regex_experiencia.search(line)
        iteracion = 1
        while(etiq_exp is not None):
            reemplazo = '<experiencia' + str(iteracion) + '>'
            line = re.sub(etiq_exp.group(),reemplazo,line)

            # Obtengo el valor de la experiencia para guardarlo en un array
            exp = regex_valor_exp.search(etiq_exp.group())
            exp = exp.group()[2:-1]
            valor_exp.append(exp)
            
            # Comprobamos si se puede reemplazar otra experiencia
            etiq_exp = regex_experiencia.search(line)
            iteracion = iteracion + 1


        return line,valor_sueldo,valor_exp

    texto4,salario,exp = preparacion(sentence)

    return texto4,salario,exp

###############################################################################

###############################################################################

def recuperacion(target,salario_orig,exp_orig):
        
        regex_salario = re.compile(r"<sal:salario.>",re.UNICODE)
        regex_exp = re.compile(r"<experiencia.>",re.UNICODE)


        # SALARIO
        salario = regex_salario.search(target)
        iteracion = 1
        while (salario is not None):

            original = "salario" + str(iteracion)
            reemplazo = str(salario_orig[iteracion - 1]) + ' euros'
            target = re.sub(original,reemplazo,target)
            
            salario = regex_salario.search(target)
            iteracion = iteracion + 1

        # EXPERIENCIA
        exp = regex_exp.search(target)
        iteracion = 1
        while (exp is not None):
            original = "experiencia" + str(iteracion)
            reemplazo = 'exp: ' + exp_orig[iteracion - 1]
            target = re.sub(original,reemplazo,target)
            
            exp = regex_exp.search(target)
            iteracion = iteracion + 1

        return target

    
    
    
    
def query(string):
    
    regex_sal = re.compile(r"(.|.\w{2})<sal:.*?>",re.UNICODE)
    regex_num_salario = re.compile(r"\d{1,}",re.UNICODE)
    regex_varios_sal = re.compile(r"\(.*?\)\sor\s\(.*?\)",re.UNICODE)
    regex_un_sal = re.compile(r"\(.*?\)",re.UNICODE)
    
    regex_exp = re.compile(r"(.|.\w{2})<exp:.*?>",re.UNICODE)
    regex_exp_valor = re.compile(r"<exp:.*?>",re.UNICODE)
    experiencias = ['No Requerida','Al menos 1 año','Al menos 2 años','Al menos 3 años','Al menos 4 años','Más de 5 años','Más de 10 años']
    
    regex_cit = re.compile(r"\s.?<cit:.*?>",re.UNICODE)
    regex_cit_nombre = re.compile(r":.*?>",re.UNICODE)
    
    regex_resto = re.compile(r"[!]{0,1}<\w{1,3}:.*?>",re.UNICODE)
    regex_resto_nombre = re.compile(r"<.*?:*?>",re.UNICODE)
    
    regex_resto_nombre
    # Modificadores de las etiquetas
    regex_modificadores = re.compile(r"gt|lt|\!|&",re.UNICODE)
    
    def buscar_modif(string):
        modificador = regex_modificadores.search(string)
        return modificador
    
    def traducir_modif(cadena):
        cadena = re.sub('gt', ' >= ', cadena)
        cadena = re.sub('lt', ' <= ', cadena)
        cadena = re.sub('!' , ' != ', cadena)
        cadena = re.sub('&', ' and ', cadena)
    
        return cadena
    

    # Añadimos un espacio al principio de la cadena para que funcione la regex
    string = ' ' + string
    
    
    # SALARIO
    salario = regex_sal.search(string)
    iteracion = 1
    while(salario is not None):
        num_salario = regex_num_salario.search(salario.group())
        
        if(iteracion == 1):
            
            string = re.sub(salario.group(),
                                '(salaryMin.value <= ' 
                                + num_salario.group() 
                                + ' and salaryMax.value >= ' 
                                + num_salario.group()
                                + ') ',string)
        elif(iteracion == 2):
            string = re.sub(salario.group(),
                                'or (salaryMin.value <= ' 
                                + num_salario.group() 
                                + ' and salaryMax.value >= ' 
                                + num_salario.group()
                                + ') ',string)
    
    
        salario = regex_sal.search(string)
        
        
        
        un_salario = regex_un_sal.search(string)
        
        # Si sólo hay una etiqueta de salario añadimos AND
        if(salario is None and iteracion == 1):
            string = string.replace(un_salario.group(),un_salario.group() + ' and')
            
        # Si hay varias etiquetas de salario añadimos paréntesis y AND
        varios_salarios = regex_varios_sal.search(string)
        if(varios_salarios is not None and iteracion == 2):
            string = string.replace(varios_salarios.group(),' (' + varios_salarios.group() + ') and')
            
        iteracion = iteracion + 1
        
        
    # EXPERIENCIA
    exp = regex_exp.search(string)
    iteracion = 1
    if(exp is not None):
        
        
        valor_exp = regex_exp_valor.search(exp.group())
        indice = experiencias.index(valor_exp.group()[6:-1])
        
        valores_filtro = experiencias[indice:]
        valores_filtro = "', '".join(valores_filtro)
        valores_filtro = "'" + valores_filtro + "'"
        
        string = re.sub(exp.group(),'experienceMin.value in ('+ valores_filtro + ') and ' ,string)
    #experienceMin.value
    
    
    # CIUDAD
    ciudad = regex_cit.search(string)
    if(ciudad is not None):
        
        nombre_ciudad = regex_cit_nombre.search(ciudad.group())
        comparador = ' = '
        modificador = buscar_modif(ciudad.group())
        
        if(modificador is not None):
            comparador = traducir_modif(modificador.group())

        string = re.sub(ciudad.group()," city" + comparador + " '" + nombre_ciudad.group()[1:-1]+ "' and ",string)
        
        
        
    # RESTO de etiquetas
    resto = regex_resto.search(string)
    
    while(resto is not None):
        
        ##modif_resto = regex_modificadores.search(resto.group())
        comparador = ' = '
        modificador = buscar_modif(resto.group())
        if(modificador is not None):
            comparador = traducir_modif(modificador.group())
            
        nombre_etiq = regex_resto_nombre.search(resto.group())
        
        filtro_resto = nombre_etiq.group()[1:-1]
        filtro_resto = re.sub(':','_',filtro_resto)
        filtro_resto = filtro_resto + comparador  +' 1 and '
        
        string = re.sub(resto.group(),filtro_resto,string)
          
        resto = regex_resto.search(string)

    query = '<query>select title, description, minRequirements, link from ofertasVN '
    if string != " ":
        query = query + 'where ' + string
        query = query[:-4]
    query += '</query>'
    return query
   

###############################################################################
###############################################################################
## Test del modelo
def decode(sentence):
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.train_dir + "/vocab_train.txt")
        fr_vocab_path = os.path.join(FLAGS.train_dir + "/vocab_target.txt")
        en_vocab, _ = initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

        # Decode from parameter.
        # Get token-ids for the input sentence.
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
            else:
                logging.warning("Sentence truncated: %s", sentence) 

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
        print("> ", end="")
        sys.stdout.flush()
    return(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
###############################################################################

###############################################################################
## Main del test del modelo
with tf.Graph().as_default():
    ##sentence = "¿Qué ofertas hay fuera de Madrid?"
    ##sentence = "¿Qué ofertas hay en Barcelona?"
    ##sentence = "¿Qué ofertas hay de java?"
    ##sentence = "Ofertas que pidan francés"
    ##sentence = "Ofertas sin java"
    ##sentence = "Ofertas sin nivel alto de inglés"
    ##sentence = "Oferta sin experiencia en bbdd y con windows ms office"
    ##sentence = "En Gran Canaria"
    ##sentence = "Muéstrame ofertas que pidan R"
    sentence = sys.argv[1]
    print (sentence)
    sentence = sentence.replace("$", " ")
    print ("SENTENCE: ")
    print (sentence)
    sentence = conversion_salario_experiencia_1(sentence)
    sentence,salario,experiencia = conversion_salario_experiencia_2(sentence)
    res = decode(sentence)
    sentence = recuperacion(res,salario,experiencia)
    sentence = query(sentence)
    print(sentence)
###############################################################################
