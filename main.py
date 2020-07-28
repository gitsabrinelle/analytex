# -*- coding: utf-8 -*-
from __future__ import division
import flask
from flask import Flask
from flask import *
import hashlib
from io import *
from io import StringIO
import string
import base64
import datetime
import random
import time
from werkzeug.utils import secure_filename
from flask import session
import itertools
import subprocess
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
import os
from collections import Counter
import operator
import spacy
from itertools import *
import scipy as s
import operator
import numpy as np
import pandas as pd
import math
from collections import Counter
import multidict as multidict
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import urllib
import requests
from os import path
import codecs
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from operator import itemgetter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyarabic.arabrepr
from collections import Counter
import re
import scipy.cluster.hierarchy as sch
from scipy.cluster import hierarchy
import pandas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app import *
from flask import Flask, request, redirect, url_for
from collections import Counter
import operator
import numpy as np
import pandas as pd

from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt
import mpld3
from mpld3._server import serve
from werkzeug.utils import secure_filename
from collections import OrderedDict
import matplotlib.pyplot as plt
import mpld3
from mpld3._server import serve
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import mysql.connector

import hashlib
from datetime import date
import os
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
import pandas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from collections import OrderedDict
from stop_words import get_stop_words
import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password="", database="analytex")
cursor = conn.cursor()
print('connecte')



"""
app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["Cache-Control"] = "no-cache, no-store, must-revalidate" # HTTP 1.1.
app.config["Pragma"] = "no-cache" # HTTP 1.0.
app.config["Expires"] = "0" # Proxies.

app.config["UPLOAD"]="static/img" 

"""



app = Flask(__name__)
UPLOAD_FOLDER = "/Users/Sabrina Nadour/Desktop/pfe22/uploads/"
ALLOWED_EXTENSIONS = set(['txt'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

arepr = pyarabic.arabrepr.ArabicRepr()


# ...................................base de donnee ..........................................................


# ...................................cookieeee ...............................................................

def cookiegenrator():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))


def update_cookie(cookie):
    sql = "UPDATE Cookie SET time='{}' WHERE cookie='{}'".format(time.time() + 10, cookie)
    cursor.execute(sql)
    conn.commit()


def check_login(cookie):
    sql = "SELECT time FROM Cookie WHERE cookie='{}'".format(cookie)
    cursor.execute(sql)
    result = cursor.fetchall()[0]
    if result < time.time():
        return redirect("/login")


def if_cookie_existe(cookie):
    sql = "SELECT cookie FROM Cookie WHERE cookie='{}'".format(cookie)
    cursor.execute(sql)
    result = cursor.fetchall()
    if result:
        return True
    else:
        return False


def Create_cookie(cookie, user_Id):
    sql = "INSERT INTO Cookie(cookie,time,id_user) VALUES('{}','{}','{}')".format(cookie, 1, user_Id)
    cursor.execute(sql)
    conn.commit()


def get_user_id(cookie):
    sql = "SELECT id_user FROM Cookie WHERE cookie='{}'".format(cookie)
    cursor.execute(sql)
    userid = cursor.fetchall()[0][0]
    return userid


def get_user_name(user_id):
    sql = "SELECT username FROM users WHERE id='{}'".format(user_id)
    cursor.execute(sql)
    username = cursor.fetchall()[0][0]
    return username


def set_cookie(cookie):
    resp = make_response(redirect("/outil1"))
    resp.set_cookie('sessionid', cookie)
    return resp


def get_Role():
    user_cookie = request.cookies.get("sessionid")
    sql = "SELECT id_user FROM Cookie WHERE cookie='{}'".format(user_cookie)
    cursor.execute(sql)
    user_id = cursor.fetchall()[0][0]
    sql = "SELECT role FROM users WHERE id='{}'".format(user_id)
    cursor.execute(sql)
    role = int(cursor.fetchall()[0][0])
    return role


def login_manager():
    if get_cookie() == None:
        return redirect("/login")
    else:
        if not if_cookie_existe(get_cookie()):
            request.cookies.pop('sessionid', None)
            return redirect("/login")
        else:
            if check_connection(get_cookie()):
                update_cookie(get_cookie())

            else:
                return redirect("/login")
    return False


def get_cookie():
    return request.cookies.get("sessionid")


def check_connection(cookie):
    current_time = time.time()
    sql = "SELECT time FROM Cookie WHERE cookie='{}'".format(cookie)
    cursor.execute(sql)
    result = cursor.fetchall()[0][0]

    if current_time > result:

        return False
    else:
        return True


# ............................................clqssification.........................................................


def normalisation_tott(text):
    t = diacritiques(text)
    t = normaliser(t)
    t = sup_mot_non_arabe(t)
    t = sup_urls(t)
    t = sup_espace(t)
    t = sup_punctuations(t)
    return t


def corpus_pers(pathcorp):
    l = {}
    src = pathcorp
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding="utf8") as inputFile:
            content = inputFile.read()
            text = normalisation_tott(content)
            text = tokenize(text)
            text = sup_stopwords(text)
            l[filename] = text
    return l


def corpus_pers2(pathcorp, listte_file_nom):
    l = {}
    src = pathcorp
    for filename in os.listdir(src):
        if filename in listte_file_nom:
            path = os.path.join(src, filename)
            with open(path, "r", encoding="utf8") as inputFile:
                content = inputFile.read()
                text = normalisation_tott(content)
                text = tokenize(text)
                text = sup_stopwords(text)
                l[filename] = text
    return l


def all_word_corpus(list_word):
    final = []
    for i in list_word:
        for j in i:
            if j not in final:
                final.append(j)
    return final


def claculer_tf(word_corp, total):
    tfidf_list = []
    for word in word_corp:
        a = afficher_TF2(word, total)
        tfidf_list.append(a)
    return tfidf_list


def claculer_tfidf(word_corp, total):
    tfidf_list = []
    for word in word_corp:
        a = afficher_TF2(word, total)

        b = afficher_IDF2(word, total)
        TF_IDFF = TF_IDF(b, a)
        tfidf_list.append(TF_IDFF)
    return tfidf_list


def distance_ecludien(list1, liste2):
    cal = 0
    dist_ecl = 0
    for i in range(0, len(list1)):
        cal = cal + ((list1[i] - liste2[i]) ** 2)
    dist_ecl = math.sqrt(cal)
    return dist_ecl


def distance_manhatan(liste1, liste2):
    return sum(abs(i - j) for i, j in zip(liste1, liste2))


def distance_jaccard(liste1, liste2):
    interaction_cardinalite = len(set.intersection(*[set(liste1), set(liste2)]))
    union_cardinalite = len(set.union(*[set(liste1), set(liste2)]))
    return interaction_cardinalite / float(union_cardinalite)


def carre(list1):
    return round(math.sqrt(sum([a * a for a in list1])), 3)


def distance_cosinus(liste1, liste2):
    num = sum(a * b for a, b in zip(liste1, liste2))
    deno = carre(liste1) * carre(liste2)
    return round((num / float(deno)), 3)


def tfidf_matrix(tfidff):
    a = [' ']
    b = np.array([])
    li_f = []
    for i in tfidff:
        a = ''
        li = []
        for j in i:
            li.append(j['TF_IDF'])
        li_f.append(li)
    lii = []
    for j in range(0, len(li_f)):
        if j == 0:
            for t in li_f[j]:
                l = []
                l.append(t)
                lii.append(l)
        else:
            for q in range(0, len(li_f[j])):
                lii[q].append((li_f[j])[q])
    l1 = lii[0]
    freq_matrix = np.array([l1])
    for i in range(1, len(lii)):
        b = np.array([lii[i]])
        freq_matrix = np.append(freq_matrix, b, axis=0)
    return freq_matrix


def CHA_manhatan(freq_matrix):
    total = []
    for i in range(0, len(freq_matrix)):
        ll = []
        for j in range(0, len(freq_matrix)):
            ll.append(round(distance_manhatan(freq_matrix[i], freq_matrix[j]), 3))
        total.append(ll)

    lis = total[0]
    distance_matrix = np.array([lis])
    for i in range(1, len(total)):
        b = np.array([total[i]])
        distance_matrix = np.append(distance_matrix, b, axis=0)
    return distance_matrix


def CHA_ecludien(freq_matrix):
    total = []
    for i in range(0, len(freq_matrix)):
        ll = []
        for j in range(0, len(freq_matrix)):
            ll.append(round(distance_ecludien(freq_matrix[i], freq_matrix[j]), 3))
        total.append(ll)

    lis = total[0]
    distance_matrix = np.array([lis])
    for i in range(1, len(total)):
        b = np.array([total[i]])
        distance_matrix = np.append(distance_matrix, b, axis=0)
    return distance_matrix


def CHA_jacard(freq_matrix):
    total = []
    for i in range(0, len(freq_matrix)):
        ll = []
        for j in range(0, len(freq_matrix)):
            ll.append(round(distance_jaccard(freq_matrix[i], freq_matrix[j]), 3))
        total.append(ll)

    lis = total[0]
    distance_matrix = np.array([lis])
    for i in range(1, len(total)):
        b = np.array([total[i]])
        distance_matrix = np.append(distance_matrix, b, axis=0)
    return distance_matrix


def CHA_cosinus(freq_matrix):
    total = []
    for i in range(0, len(freq_matrix)):
        ll = []
        for j in range(0, len(freq_matrix)):
            ll.append(round(distance_cosinus(freq_matrix[i], freq_matrix[j]), 3))
        total.append(ll)

    lis = total[0]
    distance_matrix = np.array([lis])
    for i in range(1, len(total)):
        b = np.array([total[i]])
        distance_matrix = np.append(distance_matrix, b, axis=0)
    return distance_matrix


def linkage_single(distance_matrix, nom_similarite, list_doc, dist_value):
    lis = []
    if dist_value == int(0):
        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='single')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('single')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/')
    else:

        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='single')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('single')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png')
    lis.append(pathh)

    return lis


def linkage_complete(distance_matrix, nom_similarite, list_doc, dist_value):
    lis = []
    if dist_value == int(0):
        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='complete')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('complete')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png')
    else:

        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='complete')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('complete')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png')
    lis.append(pathh)

    return lis


def linkage_average(distance_matrix, nom_similarite, list_doc, dist_value):
    lis = []
    if dist_value == int(0):
        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='average')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('average')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png')
    else:

        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='average')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('average')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png')
    lis.append(pathh)

    return lis


def linkage_ward(distance_matrix, nom_similarite, list_doc, dist_value):
    lis = []


    if dist_value == int(0):
        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='ward')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12,ax=ax,
                                     color_threshold=0)
        plt.title('ward')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotavant.png')
    else:

        pathh = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png'
        fig, ax = plt.subplots()
        threshold = dist_value  # for hline
        ax.axhline(y=threshold, c='k')
        linkage = sch.linkage(distance_matrix, method='ward')
        dendrogram4 = sch.dendrogram(linkage, labels=list_doc, leaf_rotation=90, leaf_font_size=8, p=12, ax=ax,
                                     color_threshold=0)
        plt.title('ward')
        plt.xlabel("doc")
        plt.ylabel(nom_similarite)
        # plt.show()
        lis.append(linkage)
        lis.append(linkage[-1][-2])
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/plotapres.png')
    lis.append(pathh)

    return lis

def affiche_cluster_after_cut_dondrogramme(linkage, dist_value, doc_list, path):
    fi = []
    groupes_cah = hierarchy.fcluster(linkage, dist_value, criterion='distance')
    idg = np.argsort(groupes_cah)
    liste_nom_doc = []
    liste_final_classe = []
    for i in idg:
        liste_nom_doc.append(doc_list[i])
    t = []
    listeee_doc = []
    cpt = 0
    liste_term_tfidf = {}
    grpp = list(groupes_cah[idg])
    grpp.append(0)
    for j in range(0, len(grpp)):

        if grpp[j] in t or t == [] and cpt <= len(liste_nom_doc) - 1:
            t.append(grpp[j])
            listeee_doc.append(liste_nom_doc[cpt])

        else:
            dict_info_classe = {}
            final = {}
            ki = []
            liste_term_tfidf = {}
            total2 = corpus_pers2(path, listeee_doc)
            word_corp = all_word_corpus(list(total2.values()))

            tfidff = claculer_tf(word_corp, total2)
            for p in tfidff:
                moy = 0
                term = ''
                for q in p:
                    moy = moy + q['TF']
                    term = q['term']
                moy = moy / len(listeee_doc)
                liste_term_tfidf[term] = moy
            final = dict(OrderedDict(sorted(liste_term_tfidf.items(), key=lambda x: x[1], reverse=True)))
            ki = list(final.keys())
            if cpt <= len(liste_nom_doc) - 1:
                dict_info_classe['num_classe'] = t[0]
                dict_info_classe['titre_classe'] = ki[0] + ' ' + ki[1] + ' ' + ki[2]
                dict_info_classe['nom_des_doc_de_classe'] = listeee_doc
                liste_final_classe.append(dict_info_classe)

                t = []
                t.append(grpp[j])
                listeee_doc = []
                listeee_doc.append(liste_nom_doc[cpt])

        cpt = cpt + 1
    # affichage des observations et leurs groupes
    # data=pandas.DataFrame(groupes_cah[idg], liste_nom_doc)
    # nclusts = np.unique(groupes_cah).shape[0]
    return liste_final_classe


def matrice(src):
    liss = []
    total = corpus_pers(src)
    word_corp = all_word_corpus(list(total.values()))
    list_doc = list(total.keys())
    tfidff = claculer_tfidf(word_corp, total)
    freq_matrix = tfidf_matrix(tfidff)
    liss.append(list_doc)
    liss.append(freq_matrix)
    return liss


def CHAA(src, nom_similarite, nom_linkage):
    freq_matrix = matrice(src)

    if nom_similarite == 'ecludien':
        distance_matrix = CHA_ecludien(freq_matrix[1])
        if nom_linkage == 'ward':
            b = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], 0)
        else:
            if nom_linkage == 'single':
                b = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], 0)
            else:
                if nom_linkage == 'complete':
                    b = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], 0)
                else:
                    if nom_linkage == 'average':

                        b = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], 0)

                    else:
                        print("error")
    else:
        if nom_similarite == 'cosinus':
            distance_matrix = CHA_cosinus(freq_matrix[1])
            if nom_linkage == 'ward':
                b = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], 0)
            else:
                if nom_linkage == 'single':
                    b = linkage_single(distance_matrix, nom_similarite, freq_matrix[0],0 )
                else:
                    if nom_linkage == 'complete':
                        b = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], 0)
                    else:
                        b = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], 0)

        else:
            if nom_similarite == 'jacard':
                distance_matrix = CHA_jacard(freq_matrix[1])
                if nom_linkage == 'ward':
                    b = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], 0)
                else:
                    if nom_linkage == 'single':
                        b = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], 0)
                    else:
                        if nom_linkage == 'complete':
                            b = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], 0)
                        else:
                            b = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], 0)
            else:
                if nom_similarite == 'manhatan':
                    distance_matrix = CHA_manhatan(freq_matrix[1])
                    if nom_linkage == 'ward':
                        b = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], 0)
                    else:
                        if nom_linkage == 'single':
                            b = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], 0)
                        else:
                            if nom_linkage == 'complete':
                                b= linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], 0)
                            else:
                                b = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], 0)
                else:
                    print("error")
    return b[2]


def CHAA_apres_cut_dodrogramme(src, nom_similarite, nom_linkage, value_cut):
    liss = []
    freq_matrix = matrice(src)
    if nom_similarite == 'ecludien':
        distance_matrix = CHA_ecludien(freq_matrix[1])
        if nom_linkage == 'ward':
            q = sch.linkage(distance_matrix, method='ward')
            val_max = q[-1][-2]
            pourcentage = (value_cut * val_max) / 100
            a = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)

            cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0], src)



        else:
            if nom_linkage == 'single':
                q = sch.linkage(distance_matrix, method='single')
                val_max = q[-1][-2]
                pourcentage = (value_cut * val_max) / 100
                a = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0], src)

            else:
                if nom_linkage == 'complete':
                    q = sch.linkage(distance_matrix, method='complete')
                    val_max = q[-1][-2]
                    pourcentage = (value_cut * val_max) / 100
                    a = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                    cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0], src)

                else:
                    if nom_linkage == 'average':
                        q = sch.linkage(distance_matrix, method='average')
                        val_max = q[-1][-2]
                        pourcentage = (value_cut * val_max) / 100
                        a = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                        cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                                   src)


                    else:
                        print("error")
    else:
        if nom_similarite == 'cosinus':
            distance_matrix = CHA_cosinus(freq_matrix[1])
            if nom_linkage == 'ward':
                q = sch.linkage(distance_matrix, method='ward')
                val_max = q[-1][-2]
                pourcentage = (value_cut * val_max) / 100
                a = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0], src)
            else:
                if nom_linkage == 'single':
                    q = sch.linkage(distance_matrix, method='single')
                    val_max = q[-1][-2]
                    pourcentage = (value_cut * val_max) / 100
                    a = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                    cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0], src)
                else:
                    if nom_linkage == 'complete':
                        q = sch.linkage(distance_matrix, method='complete')
                        val_max = q[-1][-2]
                        pourcentage = (value_cut * val_max) / 100
                        a = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                        cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                                   src)
                    else:
                        q = sch.linkage(distance_matrix, method='average')
                        val_max = q[-1][-2]
                        pourcentage = (value_cut * val_max) / 100
                        a = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                        cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                                   src)

        else:
            if nom_similarite == 'jacard':
                distance_matrix = CHA_jacard(freq_matrix[1])
                if nom_linkage == 'ward':
                    q = sch.linkage(distance_matrix, method='ward')
                    val_max = q[-1][-2]
                    pourcentage = (value_cut * val_max) / 100
                    a = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                    cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                               src)

                else:
                    if nom_linkage == 'single':
                        q = sch.linkage(distance_matrix, method='single')
                        val_max = q[-1][-2]
                        pourcentage = (value_cut * val_max) / 100
                        a = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                        cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                                   src)
                    else:
                        if nom_linkage == 'complete':
                            q = sch.linkage(distance_matrix, method='complete')
                            val_max = q[-1][-2]
                            pourcentage = (value_cut * val_max) / 100
                            a = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                            cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage,
                                                                                       freq_matrix[0],
                                                                                       src)
                        else:
                            q = sch.linkage(distance_matrix, method='average')
                            val_max = q[-1][-2]
                            pourcentage = (value_cut * val_max) / 100
                            a = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                            cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage,
                                                                                       freq_matrix[0],
                                                                                       src)
            else:
                if nom_similarite == 'manhatan':
                    distance_matrix = CHA_manhatan(freq_matrix[1])
                    if nom_linkage == 'ward':
                        q = sch.linkage(distance_matrix, method='ward')
                        val_max = q[-1][-2]
                        pourcentage = (value_cut * val_max) / 100
                        a = linkage_ward(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                        cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage, freq_matrix[0],
                                                                                   src)
                    else:
                        if nom_linkage == 'single':
                            q = sch.linkage(distance_matrix, method='single')
                            val_max = q[-1][-2]
                            pourcentage = (value_cut * val_max) / 100
                            a = linkage_single(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                            cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage,
                                                                                       freq_matrix[0],
                                                                                       src)
                        else:
                            if nom_linkage == 'complete':
                                q = sch.linkage(distance_matrix, method='complete')
                                val_max = q[-1][-2]
                                pourcentage = (value_cut * val_max) / 100
                                a = linkage_complete(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                                cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage,
                                                                                           freq_matrix[0],
                                                                                           src)
                            else:
                                q = sch.linkage(distance_matrix, method='average')
                                val_max = q[-1][-2]
                                pourcentage = (value_cut * val_max) / 100
                                a = linkage_average(distance_matrix, nom_similarite, freq_matrix[0], pourcentage)
                                cluster_after_cut = affiche_cluster_after_cut_dondrogramme(a[0], pourcentage,
                                                                                           freq_matrix[0],
                                                                                           src)
                else:
                    print("error")
    liss.append(cluster_after_cut)
    liss.append(a[2])
    return liss


def corpus_pers(pathcorp):
    l = {}
    src = pathcorp
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding='"utf8"') as inputFile:
            content = inputFile.read()

            text = normalisation_tott(content)
            text = tokenize(text)
            text = sup_stopwords(text)
            l[filename] = text
    return l


def all_word_corpus(list_word):
    final = []
    for i in list_word:
        for j in i:
            if j not in final:
                final.append(j)
    return final


def normalisation_tott(text):
    t = diacritiques(text)
    t = sup_mot_non_arabe(t)
    t = sup_urls(text)
    t = sup_espace(t)
    t = sup_punctuations(t)
    return t


def claculer_tfidf(word_corp, total):
    tfidf_list = []
    for word in word_corp:
        a = afficher_TF2(word, total)

        b = afficher_IDF2(word, total)
        TF_IDFF = TF_IDF(b, a)
        tfidf_list.append(TF_IDFF)
    return tfidf_list


# ...................................................................zipf....................................................................
def zipf(counts):
    a = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static'

    n = len(counts)
    ranks = range(n + 1, 1, -1)  # x-axis: the ranks
    freqs = [freq for (word, freq) in counts.items()]  # y-axis: the frequencies
    tokens = [word for (word, freq) in counts.items()]
    freqs = sorted(freqs)
    plt.loglog(ranks, freqs, marker=".", color='red')  # this plots frequency, not relative frequency
    plt.title("Zipf plot for  corpus tokens")
    plt.xlabel('"Frequency rank of token"')
    plt.ylabel('"Absolute frequency of token"')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/zipf.png')


def zipf2(file_list):
    a = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static'
    wordcount = Counter([file_list[word].lower() for word in range(len(file_list))])

    # Sort the dictionary tuples and store in list
    sorted_x = list(zip(*sorted(wordcount.items(), key=operator.itemgetter(1), reverse=True)))

    # Choosing first 100 words with highest frequency
    x = list(sorted_x[0][0:50])
    y = np.array(sorted_x[1][0:50])
    fig = plt.figure()
    df = pd.DataFrame()
    df['words'] = x
    df['frequency'] = y
    t = []
    for i in df['words']:
        artext = get_display(arabic_reshaper.reshape(i))
        t.append(artext)

    # ax = df.plot(kind='bar', title="la frequence de token de corpus")
    # ax.set_xticklabels(t, rotation=90)

    # ax2 = df.plot(ax=ax ,color="red",linewidth=2.5, linestyle="-", label="sine")
    # ax2.set_xticklabels(t, rotation=90)

    plt.ylabel("la fequence de mot")
    plt.legend(loc='upper left')
    plt.bar(x, y, color='r')

    plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/zipf.png')


# ...............................................................search motor .......................................................#
def search(req):
    try:

        final = {}
        req = sup_punctuations(req)

        e = tokenize(req)
        d = sup_stopwords(e)

        corpus = test()
        resultat1 = res(corpus.keys())

        a = afficher_IDF(d, corpus)
        r = afficher_TF(d, corpus)
        TF_IDFF = TF_IDF(a, r)
        reslt = resultat(d, TF_IDFF, resultat1)

        final = dict(OrderedDict(sorted(reslt.items(), key=lambda x: x[1], reverse=True)))

    except:

        results = "sorry we can't extraire the liste of querys suggests"
    return final

def search3(req):
    try:

        final = {}
        req = sup_punctuations(req)

        e = tokenize(req)
        d = sup_stopwords(e)

        corpus = test()
        resultat1 = res(corpus.keys())


        final = resultat1

    except:

        results = "sorry we can't extraire the liste of querys suggests"
    return final

def cor_dict(listt):
    corpus = {}
    i = 1
    for c in listt:
        z = 'doc' + str(i)
        corpus[z] = c
        i = i + 1
    return corpus


# ♦............. insialisation des scores de requette ...........#

def res(i):
    d = {}

    for a in list(i):
        d[a] = 0

    return d


# ................... calculer tf.................#

def tf(term, doc):
    l = []
    for i in doc:
        i.lower()
        l.append(i)
    return l.count(term.lower()) / float(len(l))


# ................. calculer idf..................#

def idf(term, corpus):
    num_texts_avec_term = 0
    l = []

    for i in corpus:
        l.append(i)
    for text in l:
        if term in text:
            num_texts_avec_term = num_texts_avec_term + 1

    try:
        return math.log10(float(len(l)) / num_texts_avec_term)

    except ZeroDivisionError:

        return 1.0


# ....................calculer tf_idf...............#

def tf_idf(term, doc, corpus):
    return tf(term, doc) * idf(term, corpus)


# ..............afficher le tableau d'infos avec TF...............#

def afficher_TF(QUERY_TERMS, corpus):
    info = []
    a = []
    for t in QUERY_TERMS:
        a.append(t.lower())

    for term in a:
        for doc in sorted(corpus):
            TF_TAB = {}
            TF_TAB['id_doc'] = doc
            TF_TAB['term'] = term

            TF_TAB['TF'] = tf(term, corpus[doc])

            info.append(TF_TAB)

    return info


# ..............afficher le tableau d'infos avec IDF...............#

def afficher_IDF(QUERY_TERMS, corpus):
    info = []
    a = []

    for t in QUERY_TERMS:
        a.append(t.lower())

    for term in a:
        IDF_TAB = {}
        IDF_TAB['term'] = term

        IDF_TAB['IDF'] = idf(term, corpus.values())

        info.append(IDF_TAB)

    return info


# ..............afficher le tableau d'infos avec IDF...............#

def TF_IDF(a, b):
    info = []

    for i in a:
        for j in b:
            if j['term'] == i['term']:
                TF_IDF = {}
                TF_IDF['id_doc'] = j['id_doc']
                TF_IDF['term'] = j['term']

                TF_IDF['TF_IDF'] = j['TF'] * i['IDF']
                info.append(TF_IDF)
    return list(info)


# ................ afficher les resultats des docs ..................#

def resultat(QUERY_TERMS, TF_IDF, resultat1):
    for i in QUERY_TERMS:

        for t in TF_IDF:
            if i == t['term']:
                resultat1[t['id_doc']] += t['TF_IDF']

    return resultat1


# ..............afficher le tableau d'infos avec TF...............#

def afficher_TF2(term, corpus):
    info = []
    for doc in sorted(corpus):
        TF_TAB = {}
        TF_TAB['id_doc'] = doc
        TF_TAB['term'] = term
        TF_TAB['TF'] = tf(term, corpus[doc])
        info.append(TF_TAB)

    return info


# ..............afficher le tableau d'infos avec IDF...............#

def afficher_IDF2(term, corpus):
    info = []
    IDF_TAB = {}
    IDF_TAB['term'] = term
    IDF_TAB['IDF'] = idf(term, corpus.values())
    info.append(IDF_TAB)

    return info


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


# ........................ supprimer les dialecte..........................#

def diacritiques(text):
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text


# ....................... suprimer les urls dans un text .................#

def sup_urls(string):
    regex = re.compile(r"(http|https|ftp)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    return re.sub(regex, ' ', string)


# ....................... Supprimer les nombres ..........................#

def sup_nombres(string):
    regex = re.compile(r"(\d|[\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669])+")
    return re.sub(regex, ' ', string)


# ....................... normalizer le text  ............................#

def normaliser(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return (text)


# .............. Supprimer les mots non arabes ou un symbole non arabe....#

def sup_mot_non_arabe(s):
    return ' '.join([word for word in s.split() if not re.findall(
        r'[^\s\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A]',
        word)])


# ................. Supprimer les espaces supplementaires.................#

def sup_espace(s):
    s = re.sub(r'\s+', ' ', s)
    return re.sub(r"\s{2,}", " ", s).strip()


# ................. separe la requette en mots separes ...................#

def tokenize(mot):
    mot = word_tokenize(mot)
    return mot


# ................. suprimer les mots_vides de texte......................#
def sup_stopwords1(text):
    stop_words = list(stopwords.words('arabic'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []

    for w in word_tokens:
        if w in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def sup_stopwords(listt):
    final = []
    stop = set(stopwords.words('arabic'))
    for w in listt:
        if w not in stop:
            final.append(w)

    return final


# .................. afficher les mots-vide dans le texte ..................#

def stopwords_txt(text):
    stp = []
    listt = text.split()
    stop = list(stopwords.words('arabic'))

    for i in listt:
        if i in stop:
            stp.append(i)
    return stp


# ................. suprimer remove_punctuations de texte ...............#

def sup_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def normalisation_tot(text):
    t = diacritiques(str(text))
    t = sup_urls(t)
    t = sup_espace(t)
    t = sup_punctuations(t)
    return t


def normalisation_tot3(text):
    t = diacritiques(str(text))
    t = sup_urls(t)
    t = sup_espace(t)
    return t


def normalisation_tot2(text):
    t = diacritiques(text)
    t = sup_urls(t)
    t = sup_espace(t)
    return t


def pret_tot(text):
    string = tokenize(text)
    q = sup_stopwords(string)
    return q

def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict


def makeImage(text):
   # alice_mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))

    #wc = WordCloud(font_path='arabtype.ttf',background_color="white", max_words=1000, mask=alice_mask)

    fig2 = plt.figure()
    wc = WordCloud(font_path='arial.ttf', background_color="white", max_words=1000,)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear",)
    plt.axis("off")
    #plt.show()


    #html1 = mpld3.fig_to_html(fig2)
    #serve(html1 )


    plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/nuage.png')
def nuage(text ):
    text = get_display(arabic_reshaper.reshape(text))
    makeImage(getFrequencyDictForText(text))


def file(path):
    fichier = open(path, "r", encoding='utf_8')
    content = fichier.read()
    fichier.close()
    a = []
    a.append(content)
    a.append(path)
    return a


# ...................................lire doc bibli........................................................
def test():
    l = {}
    src = '/Users/Sabrina Nadour/Desktop/pf'
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding="utf8") as inputFile:
            content = inputFile.read()

            text = sup_punctuations(content)
            text = tokenize(text)
            text = sup_stopwords(text)

            l[filename] = text
    return l


def corpus_pers():
    l = {}
    src = 'C:/Users/Sabrina Nadour/Desktop/pfe22/uploads'
    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding='"utf8"') as inputFile:
            content = inputFile.read()

            text = normalisation_tott(content)
            text = tokenize(text)
            text = sup_stopwords(text)
            l[filename] = text
    return l


def test1(nom):
    src = 'C:/Users/Sabrina Nadour/Desktop/pfe22/uploads'
    #    src=gutenberg.fileids()
    for filename in os.listdir(src):
        if filename == nom:
            path = os.path.join(src, filename)
            with open(path, "r", encoding="utf8") as inputFile:
                content = inputFile.read()
    return content


def corpus_pers(pathcorp):
    l = {}
    src = pathcorp

    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding="utf8") as inputFile:
            content = inputFile.read()

            text = sup_punctuations(content)
            text = tokenize(text)
            text = sup_stopwords(text)
            l[filename] = text
    return l


def corpus_parc(pathcorp):
    l = {}
    src = pathcorp

    for filename in os.listdir(src):
        path = os.path.join(src, filename)
        with open(path, "r", encoding="utf8") as inputFile:
            content = inputFile.read()

            text = sup_punctuations(content)
            text = tokenize(text)
            l[filename] = text
    return l


# .............................................. histogramme...........................................................
def histo(file_list, num_boutton, de, aa):
    if num_boutton == 0:
        a = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static'
        aa = int(aa)
        de = int(de)
        fig2 = plt.figure(figsize=(20, 10), dpi=80)
        sorted_x = list(zip(*sorted(file_list.items(), key=operator.itemgetter(1), reverse=True)))
        x = list(sorted_x[0][de:aa])
        print(x)
        y = np.array(sorted_x[1][de:aa])
        t = []
        df = pd.DataFrame()
        df['words'] = x
        df['frequency'] = y
        for i in df['words']:
            t.append(i)
        y_pos = np.arange(len(t))
        plt.bar(y_pos, y, color='b', align='center', alpha=0.9, label='Inter')
        plt.xticks(y_pos, tuple(x), rotation=90, fontsize=12)
        plt.ylabel('les féquences des mots', fontsize=22)
        plt.xlabel('les mots', fontsize=18)
        plt.title('histogramme ', fontsize=18)
        plt.savefig('C:/Users/Sabrina Nadour/Desktop/pfe22/static/histo.png')


    else:
        fig2 = plt.figure(figsize=(20, 10), dpi=80)
        wordcount = Counter([file_list[word].lower() for word in range(len(file_list))])
        sorted_x = list(zip(*sorted(wordcount.items(), key=operator.itemgetter(1), reverse=True)))
        x = list(sorted_x[0][0:50])
        y = np.array(sorted_x[1][0:50])
        t = []
        df = pd.DataFrame()
        df['words'] = x
        df['frequency'] = y
        for i in df['words']:
            artext = get_display(arabic_reshaper.reshape(i))
            t.append(artext)
        y_pos = np.arange(len(t))
        plt.barh(y_pos, y, align='center')
        plt.yticks(y_pos, tuple(x), rotation=90)
        plt.ylabel('les féquences des mots', fontsize=22)
        plt.xlabel('les mots', fontsize=18)
        plt.title('histogramme ', fontsize=18)
        html1 = mpld3.fig_to_html(fig2)
        serve(html1)

    return a


# groupe form

def view_concordance(reviews, word, width):
    stringtext = ' '.join(reviews)
    charindex = 0
    T = []
    while word in stringtext[charindex + len(word) + 1:]:
        charindex = stringtext.find(word, charindex + len(word) + 1)
        first = charindex - width
        last = charindex + len(word) + width + 1
        T.append(stringtext[first:last])
    return T


# end groupe form

# ................................................. ANALYSE FREQUENCIELLE .......................................................#
# ................. RECHERCHE D'UN MOT...............#
def contient(text, str):
    listt = []
    for t in text.split():
        if str in t and t not in listt:
            listt.append(t)
    return listt


def tout(text, str):
    listtout = []
    for t in text.split():
        if str == t and t not in listtout:
            listtout.append(t)
    return listtout


def start_with(text, str):
    word_start_with = []
    for t in text.split():
        if t.startswith(str) and t not in word_start_with:
            word_start_with.append(t)
    return word_start_with


def end_with(text, str):
    word_end_with = []
    for t in text.split():
        if t.endswith(str) and t not in word_end_with:
            word_end_with.append(t)
    return word_end_with


# ................. TABLE    DES    FREQUENCES...............#
def afficher_tab(text):
    string = normalisation_tot(text)
    st = pret_tot(string)
    b = count_list_mot(string)
    return b


def afficher_active(text):
    string = normalisation_tot(text)
    st = pret_tot(string)
    p = occ_listt(st)
    return p


def afficher_supp(text):
    string = normalisation_tot(text)
    w = stopwords_txt(string)
    d = occ_listt(w)
    return d


# ................. compter pour chaque mot son occ dans le texte ...............#

def count_list_mot(text):
    tf = Counter()
    cpt = 0
    for word in str(text).split():
        cpt = cpt + 1
        tf[word] += 1
    return dict(tf)


# ..................compter occ d'une liste........................................#

def occ_listt(listt):
    keywords_count = {}
    for word in listt:
        if word in keywords_count:
            keywords_count[word] += 1
        else:
            keywords_count[word] = 1

    return keywords_count


# ................. compter l'occ des mots dans le texte .........................#

def count_mots(text):
    cpt = 0
    for word in (str(text)).split():
        cpt = cpt + 1

    return cpt


# .................afficher l'occ d'un mot spec ..................................#

def affiche_occ_mot(text, mot):
    a = exist_mot(text, mot)

    if a == 1:

        t = count_list_mot(text)

        for key, values in t.items():
            if key == mot:
                return (values, mot)
    else:
        return ("le mot n'existe pas ")


# .................verifier si un mot il exist ou nn  ...............#

def exist_mot(text, mot):
    if mot in str(text):
        return 1
    else:
        return 0


# ...................nbr d'hapax .................................#

def hapx(dictt):
    a = dictt
    cpt = 0
    for i in a.values():
        if i == 1:
            cpt = cpt + 1
    return cpt


def occ(listt):
    cpt = 0
    for i in str(listt).split():
        cpt = cpt + 1
    return cpt


def forme(listt):
    cpt = 0
    form_list = []

    for i in str(listt).split():
        if i not in form_list:
            form_list.append(i)
            cpt = cpt + 1
    return cpt


def Moy_occ_par_forme(occ, form):
    moy = occ / form
    return moy


# end frequence

def afficher_statistique(text):
    a = []
    string = normalisation_tot(text)
    st = pret_tot(string)
    p = occ_listt(st)
    k = hapx(p)
    t = occ(string)
    q = forme(string)
    n = Moy_occ_par_forme(t, q)
    a.append(k)
    a.append(t)
    a.append(q)

    a.append(n)
    # return ("***** analyse frequencielle de text *******\n ## TEXT SANS LEMATISATION\nnombre d'hapax :{0}"
    # "\nnombre d'occurence: {1}\nnombre de formes: {2}"
    # "\nMoyenne d’occurrences par forme :{3}".format(k,t,q,n))
    return a


#################################################      CONCORDANCE    #############################################################

def view_concordance(stringtext, word, wi):
    tab = []
    charindex = 0
    apres = []
    q = []
    avant = []

    while word in str(stringtext[charindex + len(word) + 1:]):

        deb = charindex + len(word)
        charindex = stringtext.find(word, charindex + len(word) + 1)
        nbr = charindex + len(word)

        avant.append(stringtext[deb:charindex])
        last = charindex + len(word) + 1
        tab.append(last)
        t = str(stringtext[last:])
        print("sqbrinaaa", t)
        q.append(last)
        apres.append(t)
    avantf = []
    apresf = []
    conc = []
    for i in avant:
        m = tokenize(i)
        avantf.append(m)
    for p in apres:
        l = tokenize(p)
        apresf.append(l)

    n = ""
    i = 0
    co = []
    co2 = []
    while i in range(0, len(avantf)):
        r = avantf[i]
        co.append(r[-wi:])
        i = i + 1
    b = 0
    while b in range(0, len(apresf)):
        k = apresf[b]
        co2.append(k[0:wi])

        b = b + 1
    f = 0
    conc = []
    concfinal = []
    lstavant=[]
    lstapres = []
    all_list = []
    while f in range(0, len(apresf)):
        conc = []
        nv1 = []
        nv2 = []
        z = ' '.join(co[f])
        y = ' '.join(co2[f])
        nv1.append(z)
        nv2.append(y)
        conc.append(nv1)

        conc.append(word)
        conc.append(nv2)
        concfinal.append(conc)

        f = f + 1
    print("sqbrina52222", concfinal)
    taille = len(concfinal)
    for i in range(0, taille):
        cpt = []
        cpt = concfinal[i]
        avant = ""
        apres = ""

        for p in cpt[0]:
            avant = avant + " " + str(p)
        for q in cpt[2]:
            apres = apres + " " + str(q)
        lstavant.append(avant)
        lstapres.append(apres)

    all_list.append(lstapres)
    all_list.append(lstavant)

    return all_list


# ******************************************* 2eme concordancier *****************************************************

def tokenize(mot):
    mot = word_tokenize(mot)
    return mot


min_count = 1  # Minimum frequency of phrase to be worth keeping track of
max_n = 2  # Maximum phrase length


def get_ngram_at(sent, index, n):
    return " ".join(sent[index: index + n])


def get_concordance(all_ngrams, sentt, n, term, context, context2):
    ngram = ""

    context = int(context)
    context2 = int(context2)

    all_sentences = {}
    classement_sent = {}
    numero = 1
    all_ngrams[n] = {}
    liste1 = []
    liste2 = []
    classement = 1
    emplacement = []
    shorter_ngrams = None
    contextavant = []
    contextapres = []

    if (n > 1):
        shorter_ngrams = all_ngrams[n - 1]

    for i in range(0, len(sentt) - n + 1):

        if n > 1:
            ngram = get_ngram_at(sentt, i, n - 1)

        if (n == 1):

            ngram2 = get_ngram_at(sentt, i, n)
            ngram3 = get_ngram_at(sentt, i, n)
            classement_sent[classement] = ngram2
            classement = classement + 1
            if (ngram2 not in all_sentences):
                liste1.append(numero)
                all_sentences[ngram2] = liste1
                liste1 = []
                numero = numero + 1
            else:
                liste2 = all_sentences[ngram2]
                liste2.append(numero)
                numero = numero + 1
                all_sentences[ngram2] = liste2

        if (n == 1 or ngram in shorter_ngrams):
            ngram = get_ngram_at(sentt, i, n)

            if (ngram in all_ngrams[n]):
                all_ngrams[n][ngram] += 1
            else:
                all_ngrams[n][ngram] = 1

    for word in sentt:

        if (word == term):
            emplacement = all_sentences[term]
    for ngram in list(all_ngrams[n]):
        if all_ngrams[n][ngram] < min_count:
            del all_ngrams[n][ngram]
    lis = []
    for num in emplacement:
        tab = []

        for i in range(1, int(context + 1)):
            if (len(classement_sent)) > num + i:
                tab = []

                contextapres.append(classement_sent[num + i])

        tab.append(contextapres)

        tab.append(term)

        for i in range(1, context2 + 1):
            if (1 < num - i):
                contextavant.append(classement_sent[num - i])
        contextavant.reverse()

        tab.append(contextavant)

        lis.append(tab)

        contextapres = []
        contextavant = []

    return lis


# %%%%%%%%%%%%%%%%%%%%%%%%%%%% concordance point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_pointconcordance(all_ngrams, sentt, n, term, context, context2):
    ngram = ""

    context = int(context)
    context2 = int(context2)

    all_sentences = {}
    classement_sent = {}
    numero = 1
    all_ngrams[n] = {}
    liste1 = []
    liste2 = []
    classement = 1
    emplacement = []
    shorter_ngrams = None
    contextavant = []
    contextapres = []
    if (n > 1):
        shorter_ngrams = all_ngrams[n - 1]

    for i in range(0, len(sentt) - n + 1):

        if n > 1:
            ngram = get_ngram_at(sentt, i, n - 1)

        if (n == 1):

            ngram2 = get_ngram_at(sentt, i, n)
            ngram3 = get_ngram_at(sentt, i, n)
            classement_sent[classement] = ngram2
            classement = classement + 1
            if (ngram2 not in all_sentences):
                liste1.append(numero)
                all_sentences[ngram2] = liste1
                liste1 = []
                numero = numero + 1
            else:
                liste2 = all_sentences[ngram2]
                liste2.append(numero)
                numero = numero + 1
                all_sentences[ngram2] = liste2

        if (n == 1 or ngram in shorter_ngrams):
            ngram = get_ngram_at(sentt, i, n)

            if (ngram in all_ngrams[n]):
                all_ngrams[n][ngram] += 1
            else:
                all_ngrams[n][ngram] = 1

    for word in sentt:

        if (word == term):
            emplacement = all_sentences[term]
    for ngram in list(all_ngrams[n]):
        if all_ngrams[n][ngram] < min_count:
            del all_ngrams[n][ngram]
    lis = []
    for num in emplacement:
        ii = 1
        ii2 = 1
        tab = []
        trouve = False
        trouve2 = False
        numpos = num

        while (ii < len(sentt) and trouve != True):
            tab = []
            contextapres.append(classement_sent[num + ii])
            ii = ii + 1
            if (classement_sent[num + ii] == '.'):
                trouve = True

        tab.append(contextapres)

        tab.append(term)

        while (num - ii2 > 1 and trouve2 != True):
            contextavant.append(classement_sent[num - ii2])
            ii2 = ii2 + 1
            numpos = numpos + 1
            if ((classement_sent[num - ii2]) == '.'):
                trouve2 = True
            else:
                trouve2 = False

        contextavant.reverse()

        tab.append(contextavant)

        lis.append(tab)

        contextapres = []
        contextavant = []

    return lis


def conc2(all_ngrams, fileRead, n, term, context, context2, type):
    # fileRead = EventText.get("1.0", END)
    fileRead = normalisation_tot3(fileRead)
    # term = textentrer2.get()
    lstavant = []
    lstapres = []
    all_list = []
    if (term == ""):
        print("Error", "IL EXISTE AUCUNE term !!! \n entrer un term s'il veut plait ")
    else:
        # for i in tree4.get_children():
        # tree4.delete(i)
        sentt = tokenize(fileRead)
        if type == 1:
            for n in range(1, max_n):
                all_ngrams = get_concordance(all_ngrams, sentt, n, term, context, context2)
        else:
            for n in range(1, max_n):
                all_ngrams = get_pointconcordance(all_ngrams, sentt, n, term, context, context2)
        taille = len(all_ngrams)
        for i in range(0, taille):
            cpt = []
            cpt = all_ngrams[i]
            avant = ""
            apres = ""

            for p in cpt[0]:
                avant = avant + " " + str(p)
            for q in cpt[2]:
                apres = apres + " " + str(q)
            lstavant.append(avant)
            lstapres.append(apres)

        all_list.append(lstavant)
        all_list.append(lstapres)

    return all_list


#                                *******   END 2eme concordancier    ********


# ******************************************      N GRAMS        *****************************************************

def get_ngrams(all_ngrams, sentences, min_freq, n):
    ngram = ""
    ngram2 = ""
    all_sentences = {}
    classement_sent = {}
    numero = 1
    all_ngrams[n] = {}
    liste1 = []
    classement = 1
    shorter_ngrams = None

    if (n > 1):
        shorter_ngrams = all_ngrams[n - 1]

    for i in range(0, len(sentences) - n + 1):
        liste2 = []
        if n > 1:
            ngram = get_ngram_at(sentences, i, n - 1)

            ngram2 = get_ngram_at(sentences, i, n - 0)
            classement_sent[classement] = ngram2
            classement = classement + 1
            if (ngram2 not in all_sentences):
                liste1.append(numero)
                all_sentences[ngram2] = liste1

                liste1 = []
                numero = numero + 1
            else:
                liste2 = all_sentences[ngram2]
                liste2.append(numero)
                numero = numero + 1
                all_sentences[ngram2] = liste2

        if (n == 1):
            ngram2 = get_ngram_at(sentences, i, n)
            classement_sent[classement] = ngram2
            classement = classement + 1
            if (ngram2 not in all_sentences):
                liste1.append(numero)
                all_sentences[ngram2] = liste1
                liste1 = []
                numero = numero + 1
            else:
                liste2 = all_sentences[ngram2]
                liste2.append(numero)
                numero = numero + 1
                all_sentences[ngram2] = liste2

        if (n == 1 or ngram in shorter_ngrams):
            ngram = get_ngram_at(sentences, i, n)

            if (ngram in all_ngrams[n]):

                all_ngrams[n][ngram] += 1
            else:
                all_ngrams[n][ngram] = 1

    for ngram in list(all_ngrams[n]):
        if all_ngrams[n][ngram] < min_freq:
            del all_ngrams[n][ngram]

    return all_ngrams


def NGrams(fileRead, max_ng, min_freq):
    fileRead = normalisation_tot(fileRead)
    sentences = tokenize(fileRead)
    all_ngrams = {}
    mi = 1
    max = 5
    max_ng = int(max_ng)
    min_freq = int(min_freq)
    formes = []
    frequencess = []
    tout = []
    for n in range(1, max_ng + 1):
        all_ngrams = get_ngrams(all_ngrams, sentences, min_freq, n)
    for n in range(2, max_ng + 1):
        all_ngrams[n] = dict(OrderedDict(sorted(all_ngrams[n].items(), key=lambda x: x[1], reverse=True)))
        listng = list(all_ngrams[n].items())
        taille = len(all_ngrams[n])
        for i in range(0, taille):
            formes.append(listng[i][0])
            frequencess.append(listng[i][1])
    tout.append(formes)
    tout.append(frequencess)
    return tout


def graphe():
    x = [1, 2, 3]
    y = [2, 3, 4]
    fig1 = plt.figure()
    plt.xlabel("xlabel 1")
    plt.ylabel("ylabel 1")
    plt.title("Plot 1")
    plt.legend()
    plt.bar(x, y, label='label for bar', color='b')

    # secondgraph
    x = [1, 2, 3]
    y = [5, 3, 1]
    fig2 = plt.figure()
    plt.xlabel("xlabel 2")
    plt.ylabel("ylabel 2")
    plt.title("Plot 2")
    # plt.bar(x, y, color='r')

    # create html for both graphs
    html1 = mpld3.fig_to_html(fig1)
    html2 = mpld3.fig_to_html(fig2)
    # serve joined html to browser
    serve(html1 + html2)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/profile", methods=["GET", "POST"])
def profile():

    return render_template("profile.html")


@app.route("/header", methods=["GET", "POST"])
def header():


    return render_template("header.html")



@app.route("/paltform", methods=["GET", "POST"])
def paltform():
    return render_template("paltform.html")


@app.route("/tools", methods=["GET", "POST"])
def tools():
    return render_template("tools.html")


@app.route("/recherche", methods=["GET", "POST"])
def recherche():

    if request.method == "POST":
        req = request.form['recherche2']

        type = request.form['rech_par']
        finaldoc = []

        if type == "contenu":
            result_rech = search(req)

            for i in result_rech:
                print("je suis le chemin", i)
                if result_rech[i] != 0:
                    #if file and allowed_file(i.filename):
                       # filename = secure_filename(i.filename)

                        #i.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    path = os.path.join(app.config['UPLOAD_FOLDER'], i)
                    print("path",path)

                    sql = "INSERT INTO recherche(id_user,chemin)VALUES('{}','{}')".format(get_user_id(request.cookies.get('sessionid')),path)
                    cursor.execute(sql)
                    conn.commit()
                    finaldoc.append(i)
        if login_manager():
            return login_manager()
        sql = "SELECT * FROM recherche WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
        cursor.execute(sql)
        rech = cursor.fetchall()

        if type == "nom_doc":
            print('lui')

        return render_template("recherche.html",rech=rech)

    return render_template("recherche.html")


@app.route("/outil1", methods=["GET", "POST"])
def outil1():
    if login_manager():
        return login_manager()
    sql = "SELECT * FROM analyse WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    result1 = cursor.fetchall()
    print("toutttt",result1)
   # print("toutttt", result1[-1][8])
    im = Image.open(BytesIO(base64.b64decode(result1[-1][8])))
    print("moi",im)
    sql = "SELECT * FROM concordance WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    result2 = cursor.fetchall()

    sql = "SELECT * FROM ngrams WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    ngramss = cursor.fetchall()
   # avant_c = result2[-1][4].split('Ą')
    #apres_c = result2[-1][5].split('Ą')
    cookie = request.cookies.get("sessionid")

    if request.method == "POST":
        sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
        cursor.execute(sql)
        result = cursor.fetchall()


        if result:
            f = file(result[0][0])

            h2 = histo(afficher_tab(f[0]), 0, 0, 50)
            n2 = nuage(f[0])
            z=zipf(afficher_tab(f[0]))
            b = list(afficher_tab(f[0]).items())
            p = list(afficher_active(f[0]).items())
            d = list(afficher_supp(f[0]).items())
            a = afficher_statistique(f[0])
            l1 = len(afficher_tab(f[0]))
            l2 = len(afficher_active(f[0]))
            l3 = len(afficher_supp(f[0]))
            chemin = f[1]
            height = 500
            width = 600
            mypath = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/histo.png'
            img_file = mypath
            b64 = base64.encodestring(open(img_file, "rb").read())
            x = b64.decode('ascii')
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")

            mypath2 = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/nuage.png'
            img_file2 = mypath2
            b644 = base64.encodestring(open(img_file2, "rb").read())
            x2 = b644.decode('ascii')

            mypath3 = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/zipF.png'
            img_file3 = mypath3
            b6444 = base64.encodestring(open(img_file3, "rb").read())
            x3 = b6444.decode('ascii')





            sql = "INSERT INTO analyse(id_user,chemin,hapax,occurence,formes,occ_form,histo,nuage,zip,date) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(
                str(get_user_id(cookie)), chemin, a[0], a[1], a[2], a[3], x,x2,x3, d1)
            cursor.execute(sql)
            conn.commit()

        if request.files:

            text_file = request.files["chemin"]
            print(text_file)
            method = request.form["optradio"]
            if file and allowed_file(text_file.filename):
                filename = secure_filename(text_file.filename)

                text_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print("ne",path)


                sql = "INSERT INTO outil(path,method,id_user)VALUES('{}','{}','{}')".format(path, method, get_user_id(request.cookies.get('sessionid')))
                cursor.execute(sql)
                conn.commit()

    a = []
    for i in result2:
        x = list(i)
        x[0] = str(x[0])
        x[1] = str(x[1])
        x[2] = str(x[2])
        a.append(x)

    return render_template("outil1.html", analyse_frequencielle=result1, result2=a ,ngramss=ngramss)


@app.route("/analyse", methods=["GET", "POST"])
def analyse():
    if login_manager():
        return login_manager()
    cookie = request.cookies.get("sessionid")

    sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
    cursor.execute(sql)
    result = cursor.fetchall()
    f = file(result[-1][0])
    h2 = histo(afficher_tab(f[0]), 0, 0, 50)
    n2 = nuage(f[0])
    z=zipf(afficher_tab(f[0]))

    b = list(afficher_tab(f[0]).items())
    p = list(afficher_active(f[0]).items())
    d = list(afficher_supp(f[0]).items())
    a = afficher_statistique(f[0])
    l1 = len(afficher_tab(f[0]))
    l2 = len(afficher_active(f[0]))
    l3 = len(afficher_supp(f[0]))
    chemin = f[1]
    if request.method == "POST":

        if 'histo' in request.form:
            de = request.form["de"]
            a = request.form["a"]

            f = file(result[0][0])

            b = list(afficher_tab(f[0]).items())
            h22 = histo(afficher_tab(f[0]), 0, 0, 10)
            return render_template("analyse.html", h2=h2,n2=n2, b=b, p=p, d=d, a=a, l1=l1, l2=l2, l3=l3,
                                   chemin=chemin,h22=h22)

        if 'cher' in request.form:
            freqstart = []
            freqend = []
            freqcontient = []
            freqtout = []
            wrd = request.form["fname23"]
            option = request.form["inlineRadioOptions"]
            f = file(result[0][0])
            n2 = nuage(f[0])
            z=zipf(afficher_tab(f[0]))
            s1 = len(tout(f[0], wrd))
            s2 = len(start_with(f[0], wrd))
            s3 = len(end_with(f[0], wrd))
            s4 = len(contient(f[0], wrd))
            tout1 = tout(f[0], wrd)
            contient1 = contient(f[0], wrd)
            start_with1 = start_with(f[0], wrd)
            end_with1 = end_with(f[0], wrd)
            for i in tout(f[0], wrd):
                if i in afficher_tab(f[0]):
                    freqtout.append(afficher_tab(f[0])[i])
            for i in start_with(f[0], wrd):
                if i in afficher_tab(f[0]):
                    freqstart.append(afficher_tab(f[0])[i])
            for i in end_with(f[0], wrd):
                if i in afficher_tab(f[0]):
                    freqend.append(afficher_tab(f[0])[i])
            for i in contient(f[0], wrd):
                if i in afficher_tab(f[0]):
                    freqcontient.append(afficher_tab(f[0])[i])



            return render_template("analyse.html", h2=h2,n2=n2, b=b,z=z, p=p, d=d, a=a, l1=l1, l2=l2, l3=l3, chemin=chemin,
                                   contient1=contient1, end_with1=end_with1, start_with1=start_with1, s1=s1, s2=s2,
                                   s3=s3, s4=s4, option=str(option), freqstart=freqstart,
                                   freqend=freqend, freqcontient=freqcontient, freqtout=freqtout, tout1=tout1)

    return render_template("analyse.html", h2=h2,n2=n2,z=z, b=b, p=p, d=d, a=a, l1=l1, l2=l2, l3=l3, chemin=chemin)


@app.route("/collocation", methods=["GET", "POST"])
def collocation():
    if login_manager():
        return login_manager()
    cookie = request.cookies.get("sessionid")

    sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
    cursor.execute(sql)
    result = cursor.fetchall()
    f = file(result[-1][0])
    if request.method == "POST":
        list_coloc = []
        freqq = []
        coloc_tout = []
        coloc = request.form["coloc"]
        freqcoloc = request.form["freqcoloc"]
        NGrams_tout = NGrams(f[0], 2, freqcoloc)
        for i in range(0, len(NGrams_tout[0])):
            if coloc in NGrams_tout[0][i]:
                list_coloc.append(NGrams_tout[0][i])
                freqq.append(NGrams_tout[1][i])
        coloc_tout.append(list_coloc)
        coloc_tout.append(freqq)
        print('ma list', coloc_tout)

        return render_template("collocation.html", l3=len(coloc_tout[0]), NG=coloc_tout)
    return render_template("collocation.html")

    return render_template("collocation.html")

@app.route("/recherche/rech_analyse/<int:id>", methods=["GET", "POST"])
def rech_analyse(id):
    if login_manager():
        return login_manager()
    sql = "SELECT * FROM recherche WHERE id='{}'".format(id)
    cursor.execute(sql)
    chemin = cursor.fetchall()

    sql = "DELETE FROM recherche WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    conn.commit()

    sql = "INSERT INTO outil(path,method,id_user)VALUES('{}','{}','{}')".format(chemin[0][2],"lexicometrie", get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    conn.commit()


    sql = "SELECT * FROM analyse WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    result1 = cursor.fetchall()
    sql = "SELECT * FROM concordance WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    result2 = cursor.fetchall()

    sql = "SELECT * FROM ngrams WHERE id_user='{}'".format(get_user_id(request.cookies.get('sessionid')))
    cursor.execute(sql)
    ngramss = cursor.fetchall()

    cookie = request.cookies.get("sessionid")

    sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
    cursor.execute(sql)
    result = cursor.fetchall()

    if result:
        f = file(result[0][0])

        h2 = histo(afficher_tab(f[0]), 0, 0, 50)
        n2 = nuage(f[0])
        z = zipf(afficher_tab(f[0]))
        b = list(afficher_tab(f[0]).items())
        p = list(afficher_active(f[0]).items())
        d = list(afficher_supp(f[0]).items())
        a = afficher_statistique(f[0])
        l1 = len(afficher_tab(f[0]))
        l2 = len(afficher_active(f[0]))
        l3 = len(afficher_supp(f[0]))
        chemin = f[1]
        height = 500
        width = 600
        mypath = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/histo.png'
        img_file = mypath
        b64 = base64.encodestring(open(img_file, "rb").read())
        x = b64.decode('ascii')
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")

        mypath2 = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/nuage.png'
        img_file2 = mypath2
        b644 = base64.encodestring(open(img_file2, "rb").read())
        x2 = b644.decode('ascii')

        mypath3 = 'C:/Users/Sabrina Nadour/Desktop/pfe22/static/zipF.png'
        img_file3 = mypath3
        b6444 = base64.encodestring(open(img_file3, "rb").read())
        x3 = b6444.decode('ascii')

        sql = "INSERT INTO analyse(id_user,chemin,hapax,occurence,formes,occ_form,histo,nuage,zip,date) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(
            str(get_user_id(cookie)), chemin, a[0], a[1], a[2], a[3], x, x2, x3, d1)
        cursor.execute(sql)
        conn.commit()

    return render_template("rech_analyse.html", analyse_frequencielle=result1, result2=a ,ngramss=ngramss)



@app.route("/outil1/historique_concordance/<int:id>", methods=["GET", "POST"])
def historique_concordance(id):
    sql = "SELECT * FROM concordance WHERE id='{}'".format(id)
    cursor.execute(sql)
    result2 = cursor.fetchall()
    avant_c = result2[-1][4].split('Ą')
    apres_c = result2[-1][5].split('Ą')
    lena=len(avant_c)
    return render_template("historique_concordance.html", apres_c=apres_c, avant_c=avant_c,result2=result2,lena=lena)

@app.route("/outil1/historique_ngram/<int:id>", methods=["GET", "POST"])
def historique_ngram(id):
    sql = "SELECT * FROM ngrams WHERE id='{}'".format(id)
    cursor.execute(sql)
    result4 = cursor.fetchall()
    form = result4[-1][4].split('Ą')
    freq = result4[-1][5].split('Ą')

    lena = len(form)



    return render_template("historique_ngram.html",form=form,freq=freq, result=result4 ,lena=lena)


@app.route("/outil1/historique_analyse/<int:id>", methods=["GET", "POST"])
def historique_analyse(id):
    sql = "SELECT * FROM analyse WHERE id='{}'".format(id)
    cursor.execute(sql)
    result2 = cursor.fetchall()

    return render_template("historique_analyse.html" , analyse=result2 )


@app.route("/concordance", methods=["GET", "POST"])
def concordance():
    if login_manager():
        return login_manager()
    cookie = request.cookies.get("sessionid")

    sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
    cursor.execute(sql)
    result = cursor.fetchall()
    f = file(result[-1][0])


    if request.method == "POST":
        option = request.form["conc"]

        """uploaded_files = flask.request.files.getlist("file[]")
        filenames = []
        for fill in uploaded_files:
            if file and allowed_file(fill.filename):
                filename = secure_filename(fill.filename)
                # print('----------', filename)
                fill.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f=file(path)
                filenames.append(f[0])"""

        all_ngrams = {}
        termconc = request.form["req"]
        listconc = []
        dimension = []

        if option == 'KWAK':
            nbr = request.form["nb"]
            nbr2 = request.form["nb2"]
            affiche2 = conc2(all_ngrams, f[0], 2, termconc, int(nbr), int(nbr2), 1)
            concordance = conc2(all_ngrams, f[0], 2, termconc, int(nbr), int(nbr2), 1)
            p=view_concordance( f[0], "الخنازير", 4)
            print("conconconc",p)
            print("je dors", concordance)

            strav = ''
            strap = ''

            for i in range(0, len(concordance)):

                if i == 0:
                    a = 'Ą'.join(concordance[i])
                    strav = strav + a
                else:

                    a = 'Ą'.join(concordance[i])
                    strap = strap + a



            sql = "INSERT INTO concordance(id_user, mot, methode, avant_c, apres_c, num_avant, num_apres,date) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}')".format(
                get_user_id(request.cookies.get('sessionid')), termconc, option, strav, strap, nbr, nbr2, date.today())
            cursor.execute(sql)
            conn.commit()

            """for i in range(0,len(filenames)):
                concordance = conc2(all_ngrams, filenames[i], 2, termconc, int(nbr), int(nbr2), 1)
                dimension.append(len(concordance[0]))
                listconc.append(concordance)
            """

            return render_template("concordance.html", l1=len(affiche2[0]), term=termconc, f=f[0], av=concordance)

            # return render_template("concordance.html", l1=dimension,dim=len(listconc), term=termconc, f=f[0], av=listconc)

        if option == 'KWUT':
            nbr = request.form["nb"]
            nbr2 = request.form["nb2"]
            return render_template("concordance.html", term=termconc, f=f[0])

        if option == 'PTP':

            concordance = conc2(all_ngrams, f[0], 2, termconc, 3, 3, 2)

            strav = ''
            strap = ''

            for i in range(0, len(concordance)):

                if i == 0:
                    a = 'Ą'.join(concordance[i])
                    strav = strav + a
                else:

                    a = 'Ą'.join(concordance[i])
                    strap = strap + a

            sql = "INSERT INTO concordance(id_user, mot, methode, avant_c, apres_c, num_avant, num_apres,date) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}')".format(
                get_user_id(request.cookies.get('sessionid')), termconc, option, strav, strap, '', '', date.today())
            cursor.execute(sql)
            conn.commit()


            return render_template("concordance.html", l1=len(concordance[0]), term=termconc, f=f[0], av=concordance)

        if option == 'base_lemme':
            print("2019")
            nbr = request.form["nb"]
            nbr2 = request.form["nb2"]
            
            concordance=view_concordance( f[0], termconc, int(nbr))


            strav = ''
            strap = ''

            for i in range(0, len(concordance)):

                if i == 0:
                    a = 'Ą'.join(concordance[i])
                    strav = strav + a
                else:

                    a = 'Ą'.join(concordance[i])
                    strap = strap + a



            sql = "INSERT INTO concordance(id_user, mot, methode, avant_c, apres_c, num_avant, num_apres,date) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}')".format(
                get_user_id(request.cookies.get('sessionid')), termconc, option, strav, strap, nbr, nbr2, date.today())
            cursor.execute(sql)
            conn.commit()

            """for i in range(0,len(filenames)):
                concordance = conc2(all_ngrams, filenames[i], 2, termconc, int(nbr), int(nbr2), 1)
                dimension.append(len(concordance[0]))
                listconc.append(concordance)"""
            

            return render_template("concordance.html", l1=len(concordance[0]), term=termconc, av=concordance)

    return render_template("concordance.html", f=f[0])


@app.route("/NGram", methods=["GET", "POST"])
def NGram():
    if login_manager():
        return login_manager()
    cookie = request.cookies.get("sessionid")

    sql = "SELECT path FROM outil WHERE id_user='{}'".format(int(get_user_id(cookie)))
    cursor.execute(sql)
    result = cursor.fetchall()[0][0]

    f = file(result)
    if request.method == "POST":
        lgng = request.form["lgngram"]
        freqng = request.form["freqngram"]
        affiche3 = NGrams(f[0], lgng, freqng)
        a = ' Ą '.join(affiche3[0])
        q=[]
        for i in affiche3[1]:

            q.append(str(i))

        b= ' Ą '.join(q)

        sql = "INSERT INTO ngrams(id_user, nbr_ngram, freq_min, form, freq_form,date) VALUES ('{}','{}','{}','{}','{}','{}')".format(
            get_user_id(request.cookies.get('sessionid')), lgng, freqng, a, b,
            datetime.datetime.now())
        cursor.execute(sql)
        conn.commit()


        return render_template("NGram.html", l3=len(affiche3[0]), NG=NGrams(f[0], lgng, freqng))

    return render_template("NGram.html")




@app.route("/expl2", methods=["GET", "POST"])
def expl2():
    return render_template("expl2.html")


@app.route("/tools2", methods=["GET", "POST"])
def tools2():
    return render_template("tools2.html")


@app.route("/classification", methods=["GET", "POST"])
def classification():
    if request.method == "POST":
        if request.files:
            # text_file= request.files["txtclass"]
            simi = request.form.get("dist")
            link = request.form.get("link")
            """
            uploaded_files = flask.request.files.getlist("file[]")
            filenames = []
            for fill in uploaded_files:
                if file and allowed_file(fill.filename):
                    filename = secure_filename(fill.filename)

                    fill.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    filenames.append(filename)"""

            """if file and allowed_file(text_file.filename):
                filename = secure_filename(text_file.filename)
                #print('----------', filename)
                text_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print('pathhh', path)
            print('roseeeeeeeeeeeeee', text_file )
            print('dddddddddd', filename)
            #print('fifi', filename)"""

            if simi == '1':
                nom_similarite = 'ecludien'
            else:
                if simi == '2':

                    nom_similarite = 'manhatan'

                else:
                    if simi == '3':

                        nom_similarite = 'jacard'
                    else:
                        if simi == '4':
                            nom_similarite = 'cosinus'

            if link == '1':
                nom_linkage = 'ward'

            else:
                if link == '2':
                    nom_linkage = 'single'
                else:
                    if link == '3':
                        nom_linkage = 'complete'
                    else:
                        if link == '4':
                            nom_linkage = 'average'

            a = CHAA_apres_cut_dodrogramme('/Users/Sabrina Nadour/Desktop/test', nom_similarite, nom_linkage, 50)

            classes = []
            t = len(a[0])
            for i in a[0]:
                classes.append(list(i.values()))

        return render_template("classification.html",
                               cl=CHAA('/Users/Sabrina Nadour/Desktop/test', nom_similarite, nom_linkage), taille=t, cl2=a[1],
                               classe=classes)

    return render_template("classification.html")


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        password = str(password).encode('utf-8')
        h = hashlib.sha1()
        h.update(password)
        hashe = h.hexdigest()
        sql = "SELECT password, id FROM users WHERE username='{}'".format(username)
        cursor.execute(sql)
        info_login = cursor.fetchall()
        if info_login:
            if info_login[0][0] == hashe:
                if if_cookie_existe(get_cookie()):
                    update_cookie(get_cookie())
                else:
                    new_cookie = cookiegenrator()
                    Create_cookie(new_cookie, info_login[0][1])
                    update_cookie(new_cookie)

                    return set_cookie(new_cookie)
                return redirect("/outil1")
            else:
                return redirect("/login")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first = request.form["first"]
        last = request.form["last"]
        password = request.form["password"]
        password2 = request.form["password2"]
        mail = request.form["mail"]
        phone = request.form["txtEmpPhone"]
        if password == password2:
            password = str(password).encode('utf-8')
            h = hashlib.sha1()
            h.update(password)
            newpass = h.hexdigest()
            username = mail.split("@")[0]
            sql = "INSERT INTO users (username, first_name,last_name, password, mail, phone, role )VALUES ('{}', '{}', '{}','{}', '{}','{}','{}')".format(
                username, first, last, newpass, mail, phone, str(2))
            cursor.execute(sql)
            conn.commit()
        else:
            print("error")

    return render_template("signup.html")
@app.route("/logout", methods = ["GET","POST","DELETE"])
def logout():
    current_user_cookie=request.cookies.get("sessionid")
    sql="UPDATE Cookie SET time='{}' WHERE cookie='{}' ".format(1,current_user_cookie)
    cursor.execute(sql)
    conn.commit()
    resp = make_response(redirect("/paltform"))
    resp.set_cookie('sessionid', '', expires=0)
    return resp
    return redirect("/login")



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8888, debug=True, use_reloader=True)