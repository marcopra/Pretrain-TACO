#!/usr/bin/env python3
"""
Script per convertire uno snapshot completo di TACO Agent in un encoder checkpoint
utilizzabile con la funzione load_pretrained.

Usage:
    python convert_snapshot_to_encoder.py --snapshot_path path/to/snapshot.pt --output_path path/to/encoder.pt
    python convert_snapshot_to_encoder.py --snapshot_path path/to/snapshot.pt  # salva nella stessa cartella
"""

import argparse
import torch
from pathlib import Path
import sys


def convert_snapshot_to_encoder(snapshot_path, output_path=None):
    """
    Converte uno snapshot completo in un encoder checkpoint.
    
    Args:
        snapshot_path: Path dello snapshot da convertire
        output_path: Path dove salvare l'encoder (opzionale)
    """
    snapshot_path = Path(snapshot_path)
    
    # Verifica che il file esista
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot non trovato: {snapshot_path}")
    
    # Se output_path non è specificato, salva nella stessa cartella
    if output_path is None:
        output_path = snapshot_path.parent / f"encoder_{snapshot_path.stem}.pt"
    else:
        output_path = Path(output_path)
    
    print(f"Caricamento snapshot: {snapshot_path}")
    
    # Carica lo snapshot
    try:
        checkpoint = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Errore nel caricamento dello snapshot: {e}")
        return False
    
    print("Snapshot caricato con successo!")
    print(f"Chiavi trovate: {list(checkpoint.keys())}")
    
    # Verifica la struttura dello snapshot
    if 'agent' not in checkpoint:
        print("Errore: 'agent' non trovato nello snapshot")
        return False
    
    agent = checkpoint['agent']
    
    # Estrai i componenti dell'encoder
    encoder_state = {}
    
    # Estrai encoder
    if hasattr(agent, 'encoder') and hasattr(agent.encoder, 'state_dict'):
        encoder_state['encoder'] = agent.encoder.state_dict()
        print("✓ Encoder estratto")
    else:
        print("⚠ Encoder non trovato nell'agent")
    
    # Estrai TACO
    if hasattr(agent, 'TACO') and hasattr(agent.TACO, 'state_dict'):
        encoder_state['taco'] = agent.TACO.state_dict()
        print("✓ TACO estratto")
    else:
        print("⚠ TACO non trovato nell'agent")
    
    # Estrai action tokenizer
    if hasattr(agent, 'act_tok') and hasattr(agent.act_tok, 'state_dict'):
        encoder_state['act_tok'] = agent.act_tok.state_dict()
        print("✓ Action Tokenizer estratto")
    else:
        print("⚠ Action Tokenizer non trovato nell'agent")
    
    # Aggiungi metadati
    encoder_state['args'] = {
        'converted_from_snapshot': str(snapshot_path),
        'global_step': checkpoint.get('_global_step', 0),
        'global_episode': checkpoint.get('_global_episode', 0),
        'conversion_note': 'Converted from full agent snapshot to encoder checkpoint'
    }
    
    # Aggiungi informazioni aggiuntive se disponibili
    if '_global_step' in checkpoint:
        encoder_state['global_step'] = checkpoint['_global_step']
    if '_global_episode' in checkpoint:
        encoder_state['global_episode'] = checkpoint['_global_episode']
    
    # Verifica che almeno l'encoder sia presente
    if not encoder_state.get('encoder') and not encoder_state.get('taco'):
        print("Errore: Né encoder né TACO sono stati trovati")
        return False
    
    # Salva il checkpoint dell'encoder
    try:
        torch.save(encoder_state, output_path)
        print(f"✓ Encoder checkpoint salvato: {output_path}")
        
        # Verifica il salvataggio
        test_load = torch.load(output_path, map_location='cpu', weights_only=False)
        print(f"✓ Verifica completata. Componenti salvati: {list(test_load.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Converte uno snapshot completo in un encoder checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    # Conversione base (salva nella stessa cartella)
    python convert_snapshot_to_encoder.py --snapshot_path snapshot.pt
    
    # Conversione con path di output personalizzato
    python convert_snapshot_to_encoder.py --snapshot_path snapshot.pt --output_path encoder_converted.pt
    
    # Conversione con path assoluto
    python convert_snapshot_to_encoder.py --snapshot_path /path/to/snapshot.pt --output_path /path/to/encoder.pt
        """
    )
    
    parser.add_argument(
        '--snapshot_path', 
        type=str, 
        required=True,
        help='Path dello snapshot da convertire'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None,
        help='Path dove salvare l\'encoder (opzionale, default: stessa cartella dello snapshot)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Mostra solo cosa verrebbe convertito senza salvare'
    )
    
    args = parser.parse_args()
    
    # Informazioni iniziali
    print("="*60)
    print("CONVERSIONE SNAPSHOT -> ENCODER CHECKPOINT")
    print("="*60)
    print(f"Snapshot input: {args.snapshot_path}")
    if args.output_path:
        print(f"Output: {args.output_path}")
    else:
        snapshot_path = Path(args.snapshot_path)
        predicted_output = snapshot_path.parent / f"encoder_{snapshot_path.stem}.pt"
        print(f"Output: {predicted_output} (auto-generato)")
    print(f"Dry run: {'Sì' if args.dry_run else 'No'}")
    print("-"*60)
    
    if args.dry_run:
        print("MODALITÀ DRY RUN - Nessun file verrà salvato")
        # Carica solo per verificare la struttura
        try:
            snapshot_path = Path(args.snapshot_path)
            if not snapshot_path.exists():
                print(f"Errore: File non trovato: {snapshot_path}")
                return 1
            
            checkpoint = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            print(f"✓ Snapshot caricato con successo")
            print(f"✓ Chiavi trovate: {list(checkpoint.keys())}")
            
            if 'agent' in checkpoint:
                agent = checkpoint['agent']
                print(f"✓ Agent trovato")
                
                components = []
                if hasattr(agent, 'encoder'):
                    components.append('encoder')
                if hasattr(agent, 'TACO'):
                    components.append('TACO')
                if hasattr(agent, 'act_tok'):
                    components.append('act_tok')
                
                print(f"✓ Componenti che verrebbero estratti: {components}")
                
            print("✓ Dry run completato con successo")
            return 0
            
        except Exception as e:
            print(f"Errore durante il dry run: {e}")
            return 1
    else:
        # Esegui la conversione
        try:
            success = convert_snapshot_to_encoder(args.snapshot_path, args.output_path)
            if success:
                print("-"*60)
                print("✓ CONVERSIONE COMPLETATA CON SUCCESSO!")
                print("-"*60)
                output_path = args.output_path if args.output_path else Path(args.snapshot_path).parent / f"encoder_{Path(args.snapshot_path).stem}.pt"
                print(f"Il file encoder è ora utilizzabile con load_pretrained:")
                print(f"agent.load_pretrained('{output_path}')")
                return 0
            else:
                print("-"*60)
                print("✗ CONVERSIONE FALLITA")
                print("-"*60)
                return 1
        except Exception as e:
            print(f"Errore durante la conversione: {e}")
            return 1


if __name__ == '__main__':
    sys.exit(main())