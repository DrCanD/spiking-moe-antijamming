"""RS+CRC framing, receiver-only decoding and separate truth evaluation.

Known framing, no interleaver, no synchronization estimator, one decode/word.
Byte policy is applied on the original codeword grid, including bytes that
cross decision-block boundaries. Block refinement uses only that buffered block.
"""
import hashlib, time, zlib
import numpy as np
from coding import _codec, _decode_received, _bit_digest

N=255; K=159; DATA_BYTES=155; WORD_BITS=2040

def crc_message(data):
    data=bytes(data)
    if len(data)!=DATA_BYTES:raise ValueError('Need 155 application bytes')
    return data+(zlib.crc32(data)&0xffffffff).to_bytes(4,'big')

def crc_valid(message):
    return len(message)==K and message[-4:]==(zlib.crc32(message[:-4])&0xffffffff).to_bytes(4,'big')

def make_bits(seed,n_symbols,offset=None):
    rng=np.random.default_rng(seed)
    offset=int(rng.integers(0,WORD_BITS)) if offset is None else int(offset)
    if not 0<=offset<WORD_BITS:raise ValueError('Invalid packet phase')
    n=(n_symbols-offset)//WORD_BITS
    if n<1:raise ValueError('Stream too short')
    payloads=rng.integers(0,256,(n,DATA_BYTES),dtype=np.uint8)
    messages=np.stack([np.frombuffer(crc_message(p.tobytes()),np.uint8) for p in payloads])
    words=np.stack([np.frombuffer(bytes(_codec(K).encode(m.tobytes())),np.uint8) for m in messages])
    bits=rng.integers(0,2,n_symbols,dtype=np.uint8)
    bits[offset:offset+n*WORD_BITS]=np.unpackbits(words.reshape(-1),bitorder='big')
    meta=dict(rs_n=N,rs_k=K,application_bytes=DATA_BYTES,crc='CRC-32/ISO-HDLC via zlib.crc32, 4 bytes big-endian',
        offset_symbols=offset,n_words=n,n_symbols=n_symbols,n_coded_bits=n*WORD_BITS,
        n_application_bits=n*DATA_BYTES*8,n_filler_bits=n_symbols-n*WORD_BITS,
        tx_bits_sha256=_bit_digest(bits),payload_sha256=hashlib.sha256(payloads.tobytes()).hexdigest(),
        codewords_sha256=hashlib.sha256(words.tobytes()).hexdigest())
    return bits,payloads,words,meta

def decode_word(rx_bytes,erased):
    """Online boundary: neither original payload nor channel label enters."""
    d=_decode_received(_codec(K),np.asarray(rx_bytes,np.uint8),np.asarray(erased,bool),N-K)
    d['crc_pass']=bool(d['returned'] and crc_valid(d['payload']))
    return d

def byte_mask(symbol_flags,offset,nwords,policy):
    if policy not in ('legacy_any','block_refine2'):raise ValueError(policy)
    a=np.asarray(symbol_flags,bool)[offset:offset+nwords*WORD_BITS].reshape(nwords,N,8)
    return a.sum(axis=2)>=(1 if policy=='legacy_any' else 2)

def evaluate(soft,flags,policy,truth,payloads,original,meta,block_symbols=500,symbol_rate=1e5,transitions=(),decode_cache=None):
    """Evaluator only. CRC acceptance is computed before truth comparison."""
    soft=np.asarray(soft);flags=np.asarray(flags,bool);truth=np.asarray(truth,np.uint8)
    n=len(soft);nw=meta['n_words'];offset=meta['offset_symbols']
    if len(flags)!=n or len(truth)!=n or _bit_digest(truth)!=meta['tx_bits_sha256']:raise ValueError('Truth/length mismatch')
    if hashlib.sha256(payloads.tobytes()).hexdigest()!=meta['payload_sha256'] or hashlib.sha256(original.tobytes()).hexdigest()!=meta['codewords_sha256']:raise ValueError('Payload evidence mismatch')
    hard=(soft>0).astype(np.uint8);rx=np.packbits(hard[offset:offset+nw*WORD_BITS],bitorder='big').reshape(nw,N)
    masks=byte_mask(flags,offset,nw,policy);word_rows=[];started=time.perf_counter()
    effective=flags.copy();effective[offset:offset+nw*WORD_BITS]=np.repeat(masks,8,axis=1).reshape(-1)
    for i,(received,er,payload,word) in enumerate(zip(rx,masks,payloads,original)):
        cache_key=received.tobytes()+np.packbits(er).tobytes()
        hit=decode_cache is not None and cache_key in decode_cache
        d=decode_cache[cache_key] if hit else decode_word(received,er)
        if decode_cache is not None and not hit:decode_cache[cache_key]=d
        correct_message=bool(d['returned'] and d['payload']==crc_message(payload.tobytes()))
        application_correct=bool(d['returned'] and len(d['payload'])==K and d['payload'][:-4]==payload.tobytes())
        success=bool(d['crc_pass'] and application_correct)
        wrong=bool(d['returned'] and not correct_message)
        undetected=bool(d['crc_pass'] and not application_correct)
        v=int(er.sum());e=int(np.count_nonzero((received!=word)&~er));radius=2*e+v
        if radius<=96 and not success:raise RuntimeError('RS guaranteed-radius violation')
        start=offset+i*WORD_BITS;end=start+WORD_BITS;available=((end+block_symbols-1)//block_symbols)*block_symbols
        word_rows.append(dict(index=i,start_symbol=start,end_symbol=end,available_symbol=available,
            buffer_extra_ms=(available-end)/symbol_rate*1000,assembly_plus_buffer_ms=(available-start)/symbol_rate*1000,
            crosses_transition=any(start<int(t)<end for t in transitions),
            success=success,decoder_returned=d['returned'],decoder_attempted=d['attempted'],
            crc_pass=d['crc_pass'],crc_rejected=bool(d['returned'] and not d['crc_pass']),
            wrong_return=wrong,undetected_wrong=undetected,decoder_failure_reason=d['failure_reason'],
            erased_bytes=v,errors_outside_erasures=e,two_e_plus_v=radius,
            useful_bits=DATA_BYTES*8 if success else 0,decoder_cache_hit=hit,unique_decode_wall_s=0. if hit else d['elapsed']))
    kept=~effective;err=int(np.count_nonzero((hard!=truth)&kept));ret=int(kept.sum())
    fail=sum(not w['success'] for w in word_rows);useful=sum(w['useful_bits'] for w in word_rows)
    return dict(n_words=nw,failed_words=fail,bler=fail/nw,first_word_failure=int(not word_rows[0]['success']),
        decoder_failures=sum(not w['decoder_returned'] for w in word_rows),
        crc_rejections=sum(w['crc_rejected'] for w in word_rows),wrong_returns=sum(w['wrong_return'] for w in word_rows),
        undetected_wrong=sum(w['undetected_wrong'] for w in word_rows),over_budget_words=sum(w['erased_bytes']>96 for w in word_rows),
        useful_bits=useful,n_symbols=n,application_bits_offered=nw*DATA_BYTES*8,
        goodput_per_transmitted_symbol=useful/n,payload_delivery_fraction=useful/(nw*DATA_BYTES*8),
        crc_accepted_words=sum(w['crc_pass'] for w in word_rows),
        retained_bits=ret,retained_bit_errors=err,correct_retained_fraction=(ret-err)/n,retained_ber=err/ret if ret else None,
        evaluation_wall_s=time.perf_counter()-started,words=word_rows,
        hard_sha256=_bit_digest(hard),byte_mask_sha256=_bit_digest(masks.reshape(-1)),
        effective_mask_sha256=_bit_digest(effective)),effective
