#!/usr/bin/env python3
"""Vérifie la preuve numérique locale avec la bibliothèque standard Python.

Rejoue exactement les produits/sommes rationnels des valeurs FP32 conservées.
Ne charge aucun modèle et n'utilise ni GPU, ni NumPy, ni PyTorch. Les empreintes
des captures sources restent des références à vérifier sur les fichiers bruts
fournis dans l'archive ; ce programme valide le sous-ensemble numérique embarqué.
"""
import argparse
import base64
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct


def decode_array(item):
    if item['encoding'] != 'base64' or item['dtype'] != '<f4' or item['order'] != 'C':
        raise ValueError('Encodage FP32 inattendu')
    raw=base64.b64decode(item['data'],validate=True)
    elements=math.prod(item['shape'])
    if len(raw) != item['bytes'] or len(raw) != elements*4:
        raise ValueError('Taille de tableau incohérente')
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Empreinte du tableau incorrecte')
    values=struct.unpack('<'+'f'*elements,raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Valeur non finie')
    return values


def exact_dot(left,right):
    if len(left) != len(right):
        raise ValueError('Dimensions du produit scalaire différentes')
    return sum((Fraction.from_float(x)*Fraction.from_float(y) for x,y in zip(left,right)),Fraction(0))


def expected_rational(item):
    return Fraction(int(item['numerator']),1<<item['denominator_power_of_two'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proof',nargs='?',type=Path,
        default=Path(__file__).with_name('Preuve-numerique-routeur-M5.json'))
    args=parser.parse_args()
    raw=args.proof.read_bytes()
    proof=json.loads(raw)
    if proof['schema'] != 'airllm.public-target-router-numeric-proof.v1':
        raise ValueError('Schéma de preuve inconnu')
    arrays={name:decode_array(item) for name,item in proof['arrays'].items()}
    weights=arrays['router_weight_rows_193_104']
    if proof['arrays']['router_weight_rows_193_104']['shape'] != [2,2048]:
        raise ValueError('Dimensions des poids inattendues')
    results={}
    for label in ('h1','h2'):
        vector=arrays[label+'_router_input']
        if len(vector) != 2048:
            raise ValueError('Dimensions des activations inattendues')
        expected=proof['exact_replay'][label]
        a,b=exact_dot(vector,weights[:2048]),exact_dot(vector,weights[2048:])
        if a != expected_rational(expected['exact_dot_193']):
            raise ValueError('Produit exact de l’expert 193 incohérent')
        if b != expected_rational(expected['exact_dot_104']):
            raise ValueError('Produit exact de l’expert 104 incohérent')
        if a-b != expected_rational(expected['exact_gap_193_minus_104']) or not a>b:
            raise ValueError('Écart exact attendu incohérent')
        for bank in ('observed_gpu_logits','observed_gpu_probabilities'):
            for item in expected[bank].values():
                bits=int.from_bytes(struct.pack('<f',item['value']),'little')
                if bits != int(item['float32_bits_hex'],16):
                    raise ValueError('Bits FP32 du scalaire incohérents')
        results[label]=dict(exact_gap_193_minus_104=float(a-b),expert_193_greater_than_104=True)
    if proof['scope']['complete_full_campaign'] is not False:
        raise ValueError('Cette preuve ciblée ne termine pas la campagne complète')
    print(json.dumps(dict(status='preuve_numerique_verifiee',
        proof_sha256=hashlib.sha256(raw).hexdigest(),embedded_fp32_values=sum(map(len,arrays.values())),
        exact_rational_replay=results,model_loaded=False,gpu_used=False,
        full_campaign_completed=False),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
