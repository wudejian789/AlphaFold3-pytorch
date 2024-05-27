from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from tqdm import tqdm

class LinearNoBias(nn.Module):
    def __init__(self, inSize, outSize, hdnSize=[], ln=False):
        super(LinearNoBias, self).__init__()

        self.ln = ln
        self.layernorm = nn.LayerNorm([inSize])
        self.linears = []
        hdnSize = [inSize]+hdnSize+[outSize]
        for outSize in hdnSize[1:-1]:
            self.linears.append( nn.Linear(inSize, outSize, bias=False) )
            self.linears.append(  nn.ReLU() )
            inSize = outSize
        self.linears.append( nn.Linear(inSize, hdnSize[-1], bias=False) )
        self.linears = nn.ModuleList(self.linears)

    def forward(self, x):
        # x: ..., inSize
        if self.ln:
            x = self.layernorm(x)
        for layer in self.linears:
            x = layer(x)
        return x # ..., outSize

# Algorithm 1: Main Inference Loop
class AlphaFold3(nn.Module):
    def __init__(self, N_cycle=4, c_s=384, c_z=128, c_atom=128, c_atompair=16, c_token=384, c=64, 
                 Nte_block=2, Nmsa_block=4, Nps_block=48, Nch_block=4, N_atoms_max=4096, sigma_data=16):
        super(AlphaFold3, self).__init__()
        self.N_cycle = N_cycle

        self.input_feature_embedder = InputFeatureEmbedder(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, N_atoms_max=N_atoms_max)
        self.linear_no_bias_1 = LinearNoBias(c_token+32+32+1, c_s)
        self.linear_no_bias_2 =LinearNoBias(c_s, c_z)
        self.relative_position_encoding = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)
        self.linear_no_bias_3 = LinearNoBias(1, c_z)

        self.linear_no_bias_4 = LinearNoBias(c_z,c_z, ln=True)
        self.template_embedder = TemplateEmbedder(c_z=c_z, N_block=Nte_block, c=c)
        self.msa_module = MSAModule(c_s=c_s, c_z=c_z, c_m=c, N_block=Nmsa_block)
        self.linear_no_bias_5 = LinearNoBias(c_s,c_s, ln=True)
        self.pairformer_stack = PairformerStack(c_s,c_z, c=c, N_block=Nps_block)

        self.sample_diffusion = SampleDiffusion(c_s=c_s, c_z=c_z, c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, sigma_data=sigma_data, N_atoms_max=N_atoms_max)
        self.confidence_head = ConfidenceHead(c_s=c_s, c_z=c_z, b_pae=64, b_pde=64, b_plddt=50, b_distogram=64, N_block=Nch_block)
        # self.distogram_head = DistogramHead(c_z=c_z, b_distogram=64)
    def forward(self, fstar, T=0):
        # fstar['token_bonds_ij']: B, N_token,B_token
        fstar['residue_atom_map'] = fstar['residue_atom_indicator'].transpose(-1,-2).float()
        fstar['atom_residue_map'] = fstar['residue_atom_indicator'].float()
        B,N_token,N_atom = fstar['residue_atom_map'].shape
        dtype,device = fstar['ref_pos'].dtype,fstar['ref_pos'].device

        # should add the following part to featurize
        N_template,N_msa = 1,1
        fstar['profile'] = torch.zeros((B,N_token,32), dtype=dtype, device=device)
        fstar['deletion_mean'] = torch.zeros((B,N_token), dtype=dtype, device=device)
        fstar['template_backbone_frame_mask'] = torch.ones((B,N_template,N_token), dtype=dtype, device=device)
        fstar['template_pseudo_beta_mask'] = torch.ones((B,N_template,N_token), dtype=dtype, device=device)
        fstar['template_distogram'] = torch.zeros((B,N_template,N_token,N_token,39), dtype=dtype, device=device)
        fstar['template_unit_vector'] = torch.zeros((B,N_template,N_token,N_token,3), dtype=dtype, device=device)
        fstar['template_restype'] = torch.zeros((B,N_template,N_token), dtype=dtype, device=device)
        
        fstar['msa'] = torch.zeros((B,N_msa,N_token,32), dtype=dtype, device=device)
        fstar['has_deletion'] = torch.zeros((B,N_msa,N_token), dtype=dtype, device=device)
        fstar['deletion_value'] = torch.zeros((B,N_msa,N_token), dtype=dtype, device=device)

        # Main inference loop of AF3
        sinputs_i = self.input_feature_embedder(fstar) # B, N_token, c_token+32+32+1
        sinputs_i = self.linear_no_bias_1(sinputs_i) # B, N_token, c_s

        tmp = self.linear_no_bias_2(sinputs_i) # B, N_token, c_z
        zinit_ij = tmp[:,:,None] + tmp[:,None] # B, N_token,N_token, c_z

        zinit_ij = self.relative_position_encoding(fstar) # B, N_token,N_token, c_z
        zinit_ij = self.linear_no_bias_3(fstar['token_bonds'][...,None].float()) # B, N_token,N_token, c_z

        hz_ij,hs_i = 0,0
        for i in range(self.N_cycle):
            if i==0:
                z_ij = zinit_ij # B, N_token,N_token, c_z
            else:
                z_ij = zinit_ij + self.linear_no_bias_4(hz_ij) # B, N_token,N_token, c_z

            z_ij = z_ij+self.template_embedder(fstar, z_ij) # B, N_token,N_token, c_z
            z_ij = z_ij+self.msa_module(fstar, z_ij, sinputs_i) # B, N_token,N_token, c_z

            if i==0:
                s_i = sinputs_i # B, N_token, c_s
            else:
                s_i = sinputs_i + self.linear_no_bias_5(hs_i) # B, N_token, c_s

            s_i,z_ij = self.pairformer_stack(s_i, z_ij) # B, N_token,c_s; B, N_token,N_token,c_z

            hs_i,hz_ij = s_i,z_ij

        xpred_l,ht = self.sample_diffusion(fstar, sinputs_i, s_i, z_ij, T=T) # B, N_atom,3; B,1; 
        pplddt_l, ppae_ij, ppde_ij, presolved_l, pdistogram_ij = self.confidence_head(fstar, sinputs_i, s_i, z_ij, xpred_l)
        # pdistogram_ij = self.distogram_head(z_ij)

        return xpred_l, pplddt_l, ppae_ij, ppde_ij, presolved_l, pdistogram_ij, ht


# Algorithm 2: Construct an initial 1D embedding
class InputFeatureEmbedder(nn.Module):
    def __init__(self, c_atom=128, c_atompair=16, c_token=384, N_atoms_max=4096):
        super(InputFeatureEmbedder, self).__init__()

        self.atom_attention_encoder = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, N_atoms_max=N_atoms_max)
    def forward(self, fstar):
        # fstar['restype']: B,N_token,32; fstar['profile']: B,N_token,32; fstar['deletion_mean']: B,N_token;

        # Embed per-atom features.
        ai,_,_,_ = self.atom_attention_encoder(fstar, None,None,None) # B, N_token,c_token
        # Concatenate the per-token features.
        si = torch.cat([ai, fstar['restype'],fstar['profile'],fstar['deletion_mean'][...,None]], dim=-1) # B, N_token, c_token+32+32+1
        return si # B, N_token, c_token+32+32+1

# Algorithm 3: Relative position encoding
class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max=32, s_max=2, c_z=128):
        super(RelativePositionEncoding, self).__init__()

        self.one_hot_r_max = nn.Embedding.from_pretrained(torch.eye(2*r_max+2), freeze=True)
        self.one_hot_s_max = nn.Embedding.from_pretrained(torch.eye(2*s_max+2), freeze=True)
        self.r_max,self.s_max = r_max,s_max

        self.linear_no_bias = LinearNoBias(2*r_max+2 + 2*r_max+2 + 1 + 2*s_max+2, c_z)
    def forward(self, fstar):
        # fstar['asym_id']: B,N_token; fstar['residue_index']: B,N_token; fstar['entity_id']: B,N_token; 
        # fstar['token_index']: B,N_token; fstar['sym_id']: B,N_token;
        bsame_chain_ij = fstar['asym_id'][:,:,None]==fstar['asym_id'][:,None] # B,N_token,N_token
        bsame_residue_ij = fstar['residue_index'][:,:,None]==fstar['residue_index'][:,None] # B,N_token,N_token
        bsame_entity_ij = fstar['entity_id'][:,:,None]==fstar['entity_id'][:,None] # B,N_token,N_token

        dresidue_ij = torch.clamp(fstar['residue_index'][:,:,None]-fstar['residue_index'][:,None] + self.r_max, min=0, max=2*self.r_max) # B,N_token,N_token
        dresidue_ij[~bsame_chain_ij] = 2*self.r_max+1
        arel_pos_ij = self.one_hot_r_max(dresidue_ij.long()) # B,N_token,N_token,2*r_max+2

        dtoken_ij = torch.clamp(fstar['token_index'][:,:,None]-fstar['token_index'][:,None] + self.r_max, min=0, max=2*self.r_max) # B,N_token,N_token
        dtoken_ij[(~bsame_chain_ij)|(~bsame_residue_ij)] = 2*self.r_max+1
        arel_token_ij = self.one_hot_r_max(dtoken_ij.long()) # B,N_token,N_token,2*r_max+2

        dchain_ij = torch.clamp(fstar['sym_id'][:,:,None]-fstar['sym_id'][:,None] + self.s_max, min=0, max=2*self.s_max) # B,N_token,N_token
        dchain_ij[~bsame_chain_ij] = 2*self.s_max+1
        arel_chain_ij = self.one_hot_s_max(dchain_ij.long()) # B,N_token,N_token,2*s_max+2

        pij = self.linear_no_bias(torch.cat([arel_pos_ij, arel_token_ij, bsame_entity_ij[...,None], arel_chain_ij], dim=-1)) # B,N_token,N_token, c_z

        return pij # B,N_token,N_token, c_z

# Algorithm 4: One-hot encoding with nearest bin
# implemented this by nn.Embedding

# Algorithm 5: Atom attention encoder
class AtomAttentionEncoder(nn.Module):
    def __init__(self, c_atom, c_atompair, c_token, c_s=None, c_z=None, N_atoms_max=4096):
        super(AtomAttentionEncoder, self).__init__()

        self.linear_no_bias_1 = LinearNoBias(3+1+1+128+4*64, c_atom)
        self.linear_no_bias_2 = LinearNoBias(3, c_atompair)
        self.linear_no_bias_3 = LinearNoBias(1, c_atompair)
        self.linear_no_bias_4 = LinearNoBias(1, c_atompair)

        if c_s is not None:
            self.linear_no_bias_5 = LinearNoBias(c_s, c_atom, ln=True)
            self.linear_no_bias_6 = LinearNoBias(c_z, c_atompair, ln=True)
            self.linear_no_bias_7 = LinearNoBias(3, c_atom)

        self.linear_no_bias_8 = LinearNoBias(c_atom, c_atompair)
        self.linear_no_bias_9 = LinearNoBias(c_atompair, c_atompair, hdnSize=[c_atompair*4,c_atompair*4])

        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, N_block=3, N_head=4, N_atoms_max=N_atoms_max)

        self.linear_no_bias_10 = LinearNoBias(c_atom, c_token)
    def forward(self, fstar, r_l, strunk_i, zij):
        # fstar['ref_pos']: B, N_atom, 3; fstar['ref_charge']: B, N_atom; fstar['ref_mask']: B, N_atom; fstar['ref_element']: B, N_atom, 128; 
        # fstar['ref_atom_name_chars']: B, N_atom, 4, 64; fstar['ref_space_uid']: B, N_atom, 1
        B,N_atom = fstar['ref_atom_name_chars'].shape[:2]

        # Create the atom single conditioning: Embed per-atom meta data
        c_l = self.linear_no_bias_1( torch.cat([fstar['ref_pos'], \
                                                fstar['ref_charge'][...,None], \
                                                fstar['ref_mask'][...,None], \
                                                fstar['ref_element'], \
                                                fstar['ref_atom_name_chars'].reshape(B,N_atom,-1)],dim=2) ) # B, N_atom, c_atom

        # Embed offsets between atom reference positions
        d_lm = fstar['ref_pos'][...,None,:] - fstar['ref_pos'][:,None] # B,N_atom,N_atom,3
        v_lm = fstar['ref_space_uid'][...,:,None]==fstar['ref_space_uid'][...,None] # B,N_atom,N_atom
        p_lm = self.linear_no_bias_2( d_lm ) * v_lm[...,None].float() # B,N_atom,N_atom,c_atompair

        # Embed pairwise inverse squared distances, and the valid mask.
        p_lm = p_lm+self.linear_no_bias_3( 1 / (1+torch.norm(d_lm,dim=-1,keepdim=True)) ) * v_lm[...,None].float() # B, N_atom,N_atom,c_atompair
        p_lm = p_lm+self.linear_no_bias_4( v_lm[...,None].float() ) * v_lm[...,None].float() # B, N_atom,N_atom,c_atompair

        # Initialize the atom single representation as the single conditioning.
        q_l = c_l # B, N_atom, c_atom
        # If provided, add trunk embeddings and noisy positions.
        if r_l is not None:
            # r_l: B, N_atom, 3
            # strunk_i: B, N_token, c_s
            # zij: B, N_token,N_token, c_z

            # Broadcast the single and pair embedding from the trunk.
            c_l = c_l+fstar['atom_residue_map']@self.linear_no_bias_5(strunk_i) # B, N_atom, c_atom
            p_lm = p_lm+(fstar['atom_residue_map'][:,None]@self.linear_no_bias_6( zij ).permute(0,3,1,2)@fstar['residue_atom_map'][:,None]).permute(0,2,3,1) # B, N_atom,N_atom, c_atompair

            # Add the noisy positions.
            q_l = q_l+self.linear_no_bias_7(r_l) # B, N_atom, c_atom

        # Add the combined single conditioning to the pair representation.
        c_l_ = self.linear_no_bias_8(F.relu(c_l)) # B, N_atom, c_atompair
        p_lm = p_lm + (c_l_[...,None,:] + c_l_[:,None]) # B, N_atom,N_atom, c_atompair

        # Run a small MLP on the pair activations
        p_lm = p_lm+self.linear_no_bias_9(F.relu(p_lm)) # B, N_atom,N_atom, c_atompair

        # Cross attention transformer.
        q_l = self.atom_transformer(q_l, c_l, p_lm) # B, N_atom, c_atom

        # Aggregate per-atom representation to per-token representation
        q_l_ = F.relu(self.linear_no_bias_10(q_l)) # B, N_atom, c_token
        # fstar['residue_atom_map']: B, N_token,N_atom
        a_i = (fstar['residue_atom_map'] @ q_l_) / fstar['residue_atom_map'].sum(axis=-1,keepdims=True) # B, N_token,c_token

        return a_i, q_l, c_l, p_lm # B,N_token,c_token; B,N_atom,c_atom; B,N_atom,c_atom; B,N_atom,N_atom,c_atompair

# Algorithm 6: Atom attention decoder
class AtomAttentionDecoder(nn.Module):
    def __init__(self, c_atom, c_atompair, c_token, N_atoms_max=4096):
        super(AtomAttentionDecoder, self).__init__()

        self.linear_no_bias_1 = LinearNoBias(c_token, c_atom)
        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, N_block=3, N_head=4, N_atoms_max=N_atoms_max)
        self.linear_no_bias_2 = LinearNoBias(c_atom, 3, ln=True)
    def forward(self, fstar, a_i, q_l, c_l, p_lm):
        # a_i: B, N_token,c_token; q_l: B, N_atom,c_atom; c_l: B, N_atom,c_atom; p_lm: B, N_atom,N_atom, c_atompair
        # fstar['atom_residue_map']: B, N_atom, N_residue

        # Broadcast per-token activations to per-atom activations and add the skip connection
        q_l = fstar['atom_residue_map']@self.linear_no_bias_1(a_i) + q_l # B, N_atom, c_atom

        # Cross attention transformer. 
        q_l = self.atom_transformer(q_l, c_l, p_lm) # B, N_atom, c_atom

        # Map to positions update.
        r_l = self.linear_no_bias_2(q_l) # B, N_atom, 3
        
        return r_l # B, N_atom, 3

# Algorithm 7: Atom Transformer
class AtomTransformer(nn.Module):
    def __init__(self, c_atom, c_atompair, N_block=3, N_head=4, N_queries=32, N_keys=128, N_atoms_max=4096):
        super(AtomTransformer, self).__init__()

        S_subsetcentres=[15.5+i*32 for i in range(int((N_atoms_max-15.5)//32+1))]
        c = torch.tensor(S_subsetcentres, dtype=torch.float32)
        # sequence-local atom attention is equivalent to self attention within rectangular blocks along the diagonal.
        print('initializing the magic beta_ij...')
        beta_lm = [[0 if ((torch.abs(l-c) < N_queries/2) & \
                         (torch.abs(m-c) < N_keys/2) ).any() 
                      else -10**10 for m in range(N_atoms_max)] for l in tqdm(range(N_atoms_max))]
        self.beta_lm = nn.Parameter(torch.tensor(beta_lm, dtype=torch.float32)[None], requires_grad=False) # 1, N_atom, N_atom

        self.diffusion_transformer = DiffusionTransformer(c_a=c_atom, c_s=c_atom, c_z=c_atompair, N_block=N_block, N_head=N_head)
    def forward(self, q_l, c_l, p_lm):
        # q_l: B, N_atom,c_atom; c_l: B, N_atom,c_atom; p_lm: B, N_atom,N_atom, c_atompair
        N_atom = q_l.shape[1]
        q_l = self.diffusion_transformer(q_l, c_l, p_lm, self.beta_lm[:,:N_atom,:N_atom]) # B, N_atom,c_atom
        
        return q_l # B, N_atom,c_atom

# Algorithm 8: MSA Module
class MSAModule(nn.Module):
    def __init__(self, c_s, c_z=128, c_m=64, N_block=4):
        super(MSAModule, self).__init__()

        self.N_block = N_block

        self.linear_no_bias_1 = LinearNoBias(34, c_m)
        self.linear_no_bias_2 = LinearNoBias(c_s, c_m)

        self.outer_product_mean_list = nn.ModuleList([OuterProductMean(c_m=c_m, c=32, c_z=c_z) for i in range(N_block)])
        self.msa_pair_weighted_averaging_list = nn.ModuleList([MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, c=8, N_head=8) for i in range(N_block)])
        self.transition_list_1 = nn.ModuleList([Transition(c=c_m, n=4) for i in range(N_block)])

        self.triangle_multiplication_outgoing_list = nn.ModuleList([TriangleMultiplicationOutgoing(c_z=c_z,c=128) for i in range(N_block)])
        self.triangle_multiplication_incoming_list = nn.ModuleList([TriangleMultiplicationIncoming(c_z=c_z,c=128) for i in range(N_block)])
        self.triangle_attention_starting_node_list = nn.ModuleList([TriangleAttentionStartingNode(c_z=c_z, c=32, N_head=4) for i in range(N_block)])
        self.triangle_attention_ending_node_list = nn.ModuleList([TriangleAttentionEndingNode(c_z=c_z, c=32, N_head=4) for i in range(N_block)])

        self.transition_list_2 = nn.ModuleList([Transition(c=c_z, n=4) for i in range(N_block)])

    def forward(self, fstar, z_ij, sinputs_i):
        # fstar['msa']: B, N_msa, N_token, 32; fstar['has_deletion']: B, N_msa,N_token; fstar['deletion_value']: B, N_msa,N_token;
        # z_ij: B, N_token,N_token, c_z; sinputs_i: B, N_token, c_s
        m_si = torch.cat([fstar['msa'],fstar['has_deletion'][...,None],fstar['deletion_value'][...,None]], dim=-1) # B, N_msa,N_token, 34

        # s = SampleRandomWithoutReplacement(...) # warning, don't know is this used for

        m_si = self.linear_no_bias_1(m_si) # B, N_msa,N_token, c_m
        m_si = m_si+self.linear_no_bias_2(sinputs_i) # B, N_msa,N_token, c_m

        for outer_product_mean,msa_pair_weighted_averaging,transition_1,\
            triangle_multiplication_outgoing,triangle_multiplication_incoming,\
            triangle_attention_starting_node,triangle_attention_ending_node,transition_2 in zip(self.outer_product_mean_list,self.msa_pair_weighted_averaging_list,self.transition_list_1,\
                                                                                                self.triangle_multiplication_outgoing_list,self.triangle_multiplication_incoming_list,\
                                                                                                self.triangle_attention_starting_node_list,self.triangle_attention_ending_node_list,\
                                                                                                self.transition_list_2):
            # Communication
            z_ij = z_ij+outer_product_mean(m_si) # B, N_token,N_token, c_z

            # MSA stack
            m_si = m_si+F.dropout(msa_pair_weighted_averaging(m_si, z_ij), p=0.15) # B,N_msa, N_token,c_m

            m_si = m_si+transition_1(m_si) # B,N_msa, N_token,c_m

            # Pair stack
            z_ij = z_ij+F.dropout(triangle_multiplication_outgoing(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+F.dropout(triangle_multiplication_incoming(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+F.dropout(triangle_attention_starting_node(z_ij), p=0.25) # B, N_token,N_otken, c_z
            z_ij = z_ij+F.dropout(triangle_attention_ending_node(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+transition_2(z_ij) # B, N_token,N_token, c_z

        return z_ij # B, N_token,N_token, c_z

# Algorithm 9: Outer product mean
class OuterProductMean(nn.Module):
    def __init__(self, c_m, c=32, c_z=128):
        super(OuterProductMean, self).__init__()

        self.c = c

        self.ln = nn.LayerNorm([c_m])
        self.linear_no_bias = LinearNoBias(c_m, c*2)
        self.linear = nn.Linear(c*c, c_z)

    def forward(self, m_si):
        # m_si: B, N_msa,N_token, c_m
        m_si = self.ln(m_si) # B, N_msa,N_token, c_m

        ab_si = self.linear_no_bias(m_si) # B, N_msa,N_token, c*2
        a_si,b_si = ab_si[...,:self.c],ab_si[...,self.c:] # B, N_msa,N_token, c

        o_ij = (a_si[...,None]*b_si[...,None,:]).mean(dim=1).flatten(start_dim=-2) # B, N_token,N_token, c*c; warning! not sure for this part.
        z_ij = self.linear(o_ij) # B, N_token,N_token, c_z

        return z_ij # B, N_token,N_token, c_z

# Algorithm 10: MSA pair weighted averaging with gating
class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, c_m, c_z, c=32, N_head=8):
        super(MSAPairWeightedAveraging, self).__init__()

        self.c = c
        self.N_head = N_head

        self.ln = nn.LayerNorm([c_m])
        self.linear_no_bias_1 = LinearNoBias(c_m, c*N_head)
        self.linear_no_bias_2 = LinearNoBias(c_z, N_head, ln=True)
        self.linear_no_bias_3 = LinearNoBias(c_m, c*N_head)

        self.linear_no_bias_4 = LinearNoBias(c*N_head, c_m)

    def forward(self, m_si, z_ij):
        # m_si: B, N_msa,N_token, c_m; z_ij: B, N_token,N_token, c_z
        B,N_msa,N_token,c_m = m_si.shape

        # Input projections
        m_si = self.ln(m_si) # B, N_msa,N_token, c_m
        vh_si = self.linear_no_bias_1(m_si).reshape(B,N_msa,N_token,self.N_head,self.c).transpose(-2,-3) # B, N_msa,N_head, N_token,c
        bh_ij = self.linear_no_bias_2(z_ij).reshape(B,N_token,N_token,self.N_head).permute(0,3,1,2) # B,N_head, N_token,N_token
        gh_si = F.sigmoid(self.linear_no_bias_3(m_si)).reshape(B,N_msa,N_token,self.N_head,self.c).transpose(-2,-3) # B, N_msa,N_head, N_token,c

        # Weighted average with gating
        wh_ij = F.softmax(bh_ij, dim=-2) # B,N_head, N_token,N_token

        oh_si = gh_si * (wh_ij[:,None]@vh_si) # B,N_msa, N_head, N_token,c

        # Output projection
        m_si = self.linear_no_bias_4(oh_si.transpose(2,3).reshape(B,N_msa, N_token,-1)) # B,N_msa, N_token,c_m
        
        return m_si # B,N_msa, N_token,c_m

# Algorithm 11: Transition layer
class Transition(nn.Module):
    def __init__(self, c, n=4):
        super(Transition, self).__init__()

        self.ln = nn.LayerNorm([c])
        self.linear_no_bias_1 = LinearNoBias(c, n*c)
        self.linear_no_bias_2 = LinearNoBias(c, n*c)
        self.linear_no_bias_3 = LinearNoBias(n*c, c)

    def forward(self, x):
        # x: B,..., c
        x = self.ln(x) # B,..., c
        a = self.linear_no_bias_1(x) # B,..., n*c
        b = self.linear_no_bias_2(x) # B,..., n*c
        x = self.linear_no_bias_3( F.silu(a)*b ) # B,..., c
        
        return x # B,..., c

# Algorithm 12: Triangular multiplicative update using "outgoing" edges
class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, c_z, c=128):
        super(TriangleMultiplicationOutgoing, self).__init__()

        self.c = c

        self.ln = nn.LayerNorm([c_z])

        self.linear_no_bias_1 = LinearNoBias(c_z, c*2)
        self.linear_no_bias_2 = LinearNoBias(c_z, c*2)

        self.linear_no_bias_3 = LinearNoBias(c_z, c_z)

        self.linear_no_bias_4 = LinearNoBias(c, c_z, ln=True)

    def forward(self, z_ij):
        # z_ij: B, N_token,N_token, c_z
        z_ij = self.ln(z_ij) # B, N_token,N_token, c_z

        ab_ij = F.sigmoid(self.linear_no_bias_1(z_ij)) * self.linear_no_bias_2(z_ij) # B, N_token,N_token, c*2
        a_ij,b_ij = ab_ij[...,:self.c],ab_ij[...,self.c:] # B, N_token,N_token, c
        g_ij = F.sigmoid(self.linear_no_bias_3(z_ij)) # B, N_token,N_token, c_z

        z_ij = g_ij * self.linear_no_bias_4( (a_ij.permute(0,3,1,2)@b_ij.permute(0,3,2,1)).permute(0,2,3,1) ) # B, N_token,N_token, c_z
        
        return z_ij # B, N_token,N_token, c_z

# Algorithm 13: Triangular multiplicative update using "incoming" edges
class TriangleMultiplicationIncoming(nn.Module):
    def __init__(self, c_z, c=128):
        super(TriangleMultiplicationIncoming, self).__init__()

        self.c = c

        self.ln = nn.LayerNorm([c_z])

        self.linear_no_bias_1 = LinearNoBias(c_z, c*2)
        self.linear_no_bias_2 = LinearNoBias(c_z, c*2)

        self.linear_no_bias_3 = LinearNoBias(c_z, c_z)

        self.linear_no_bias_4 = LinearNoBias(c, c_z, ln=True)

    def forward(self, z_ij):
        # z_ij: B, N_token,N_token, c_z
        z_ij = self.ln(z_ij) # B, N_token,N_token, c_z

        ab_ij = F.sigmoid(self.linear_no_bias_1(z_ij)) * self.linear_no_bias_2(z_ij) # B, N_token,N_token, c*2
        a_ij,b_ij = ab_ij[...,:self.c],ab_ij[...,self.c:] # B, N_token,N_token, c
        g_ij = F.sigmoid(self.linear_no_bias_3(z_ij)) # B, N_token,N_token, c_z

        z_ij = g_ij * self.linear_no_bias_4( (a_ij.permute(0,3,2,1)@b_ij.permute(0,3,1,2)).permute(0,2,3,1) ) # B, N_token,N_token, c_z
        
        return z_ij # B, N_token,N_token, c_z

# Algorithm 14: Triangle gated self-attention around starting node
class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, c_z, c=32, N_head=4):
        super(TriangleAttentionStartingNode, self).__init__()
        
        self.c = c
        self.N_head = N_head

        self.ln = nn.LayerNorm([c_z])
        self.linear_no_bias_1 = LinearNoBias(c_z, c*N_head*3)
        self.linear_no_bias_2 = LinearNoBias(c_z, N_head)
        self.linear_no_bias_3 = LinearNoBias(c_z, c*N_head)

        self.linear_no_bias_4 = LinearNoBias(c*N_head, c_z)

    def forward(self, z_ij):
        # z_ij: B, N_token,N_token, c_z
        B,N_token,_,c_z = z_ij.shape

        # input projections
        z_ij = self.ln(z_ij) # B, N_token,N_token, c_z

        qkvh_ij = self.linear_no_bias_1(z_ij) # B, N_token,N_token, c*N_head*3
        qh_ij,kh_ij,vh_ij = qkvh_ij[...,:self.c*self.N_head].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4), \
                            qkvh_ij[...,self.c*self.N_head:self.c*self.N_head*2].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4), \
                            qkvh_ij[...,self.c*self.N_head*2:self.c*self.N_head*3].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4) # B, N_head, N_token,N_token, c
        
        bh_ij = self.linear_no_bias_2(z_ij).reshape(B,N_token,N_token,self.N_head).permute(0,3,1,2) # B, N_head, N_token,N_token
        gh_ij = F.sigmoid(self.linear_no_bias_3(z_ij)).reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4) # B, N_head, N_token,N_token, c

        # Attention
        ah_ijk = F.softmax(1/np.sqrt(self.c) * (qh_ij @ kh_ij.transpose(-1,-2)) + bh_ij[:,:,None], dim=-1) # B, N_head, N_token, N_token,N_token
        oh_ij = gh_ij * (ah_ijk@vh_ij) # B, N_head, N_token, N_token,c

        # Output projection
        z_ij = self.linear_no_bias_4(oh_ij.permute(0,2,3,1,4).reshape(B,N_token,N_token,-1)) # B, N_token,B_token,c_z

        return z_ij # B, N_token,B_token,c_z

# Algorithm 15: Triangle gated self-attention around ending node
class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, c_z, c=32, N_head=4):
        super(TriangleAttentionEndingNode, self).__init__()
        
        self.c = c
        self.N_head = N_head

        self.ln = nn.LayerNorm([c_z])
        self.linear_no_bias_1 = LinearNoBias(c_z, c*N_head*3)
        self.linear_no_bias_2 = LinearNoBias(c_z, N_head)
        self.linear_no_bias_3 = LinearNoBias(c_z, c*N_head)

        self.linear_no_bias_4 = LinearNoBias(c*N_head, c_z)

    def forward(self, z_ij):
        # z_ij: B, N_token,N_token, c_z
        B,N_token,_,c_z = z_ij.shape

        # input projections
        z_ij = self.ln(z_ij) # B, N_token,N_token, c_z

        qkvh_ij = self.linear_no_bias_1(z_ij) # B, N_token,N_token, c*N_head*3
        qh_ij,kh_ij,vh_ij = qkvh_ij[...,:self.c*self.N_head].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4), \
                            qkvh_ij[...,self.c*self.N_head:self.c*self.N_head*2].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4), \
                            qkvh_ij[...,self.c*self.N_head*2:self.c*self.N_head*3].reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4) # B, N_head, N_token,N_token, c
        
        bh_ij = self.linear_no_bias_2(z_ij).reshape(B,N_token,N_token,self.N_head).permute(0,3,1,2) # B, N_head, N_token,N_token
        gh_ij = F.sigmoid(self.linear_no_bias_3(z_ij)).reshape(B,N_token,N_token,self.N_head,self.c).permute(0,3,1,2,4) # B, N_head, N_token,N_token, c

        # Attention
        ah_ijk = F.softmax(1/np.sqrt(self.c) * (qh_ij @ kh_ij.transpose(-2,-3).transpose(-1,-2)) + bh_ij.transpose(-1,-2)[:,:,:,None], dim=-1) # B, N_head, N_token, N_token,N_token
        oh_ij = gh_ij * (ah_ijk@vh_ij.transpose(2,3)) # B, N_head, N_token, N_token,c

        # Output projection
        z_ij = self.linear_no_bias_4(oh_ij.permute(0,2,3,1,4).reshape(B,N_token,N_token,-1)) # B, N_token,B_token,c_z

        return z_ij # B, N_token,B_token,c_z

# Algorithm 16: Template embedder
class TemplateEmbedder(nn.Module):
    def __init__(self, c_z, N_block=2, c=64):
        super(TemplateEmbedder, self).__init__()

        self.linear_no_bias_1 = LinearNoBias(c_z, c, ln=True)
        self.linear_no_bias_2 = LinearNoBias(39+1+3+1+2, c)

        self.pairformer_stack = PairformerStack(c_s=None, c_z=c, c=c, N_block=N_block)
        self.ln = nn.LayerNorm([c])

        self.linear_no_bias_3 = LinearNoBias(c,c_z)

    def forward(self, fstar, z_ij):
        # fstar['template_backbone_frame_mask']: B, N_template, N_token
        # fstar['template_pseudo_beta_mask']: B, N_template, N_token
        
        # fstar['template_distogram']: B, N_template, N_token,N_token, 39
        
        # fstar['template_unit_vector']: B, N_template, N_token,N_token, 3
        # fstar['template_restype']: B, N_template, N_token;
        # fstar['asym_id']: B, N_token; 
        # z_ij: B, N_token,N_token, c_z

        _,N_template,_ = fstar['template_backbone_frame_mask'].shape
        B,N_token,_,c_z = z_ij.shape

        btemplate_backbone_frame_mask_ij = fstar['template_backbone_frame_mask'][...,None].bool() & fstar['template_backbone_frame_mask'][:,:,None].bool() # B, N_template, N_token,N_token
        btemplate_pseudo_beta_mask_ij = fstar['template_pseudo_beta_mask'][...,None].bool() & fstar['template_pseudo_beta_mask'][:,:,None].bool() # B, N_template, N_token,N_token

        a_tij = torch.cat([fstar['template_distogram'], \
                           btemplate_backbone_frame_mask_ij[...,None], \
                           fstar['template_unit_vector'], \
                           btemplate_pseudo_beta_mask_ij[...,None]], dim=4) # B, N_template, N_token,N_token, 39+1+3+1

        a_tij = a_tij * (fstar['asym_id'][...,None]==fstar['asym_id'][:,None])[:,None,...,None].float() # B, N_template, N_token,N_token, 39+1+3+1
        a_tij = torch.cat([a_tij, fstar['template_restype'][...,None,None].repeat(1,1,1,N_token,1),\
                                  fstar['template_restype'][:,:,None,...,None].repeat(1,1,N_token,1,1)], dim=-1) # B, N_template, N_token,N_token, 39+1+3+1+2
        
        u_ij = 0
        for t in range(N_template):
            v_ij = self.linear_no_bias_1(z_ij) + self.linear_no_bias_2(a_tij[:,t]) # B, N_token,N_token, c
            v_ij = v_ij+self.pairformer_stack(None, v_ij)[1] # B, N_token,N_token, c
            u_ij = u_ij+self.ln(v_ij) # B, N_token,N_token, c

        u_ij = u_ij / N_template
        u_ij = self.linear_no_bias_3(u_ij) # B, N_token,N_token, c

        return u_ij # B, N_token,N_token, c_z

# Algorithm 17: Pairformer stack
class PairformerStack(nn.Module):
    def __init__(self, c_s, c_z, c, N_block=48):
        super(PairformerStack, self).__init__()

        self.triangle_multiplication_outgoing_list = nn.ModuleList([TriangleMultiplicationOutgoing(c_z, c) for i in range(N_block)])
        self.triangle_multiplication_incoming_list = nn.ModuleList([TriangleMultiplicationIncoming(c_z, c) for i in range(N_block)])
        self.triangle_attention_starting_node_list = nn.ModuleList([TriangleAttentionStartingNode(c_z, c) for i in range(N_block)])
        self.triangle_attention_ending_node_list = nn.ModuleList([TriangleAttentionEndingNode(c_z, c) for i in range(N_block)])

        self.transition_list_1 = nn.ModuleList([Transition(c_z) for i in range(N_block)])
        if c_s is not None:
            self.attention_pair_bias_list = nn.ModuleList([AttentionPairBias(c_s, None, c_z, N_head=16) for i in range(N_block)])
            self.transition_list_2 = nn.ModuleList([Transition(c_s) for i in range(N_block)])
        else:
            self.attention_pair_bias_list = nn.ModuleList([None for i in range(N_block)])
            self.transition_list_2 = nn.ModuleList([None for i in range(N_block)])

    def forward(self, s_i, z_ij):
        # s_i: B, N_token, c_s; z_ij: B, N_token,N_token, c_z
        for triangle_multiplication_outgoing,triangle_multiplication_incoming,\
            triangle_attention_starting_node,triangle_attention_ending_node,\
            transition_1,attention_pair_bias,transition_2 in zip(self.triangle_multiplication_outgoing_list,self.triangle_multiplication_incoming_list,\
                                                                 self.triangle_attention_starting_node_list,self.triangle_attention_ending_node_list,\
                                                                 self.transition_list_1,self.attention_pair_bias_list,self.transition_list_2):
            
            z_ij = z_ij+F.dropout(triangle_multiplication_outgoing(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+F.dropout(triangle_multiplication_incoming(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+F.dropout(triangle_attention_starting_node(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+F.dropout(triangle_attention_ending_node(z_ij), p=0.25) # B, N_token,N_token, c_z
            z_ij = z_ij+transition_1(z_ij) # B, N_token,N_token, c_z
            
            if s_i is not None:
                s_i = s_i+attention_pair_bias(s_i, None, z_ij, beta_ij=0.0) # B, N_token, c_s
                s_i = s_i+transition_2(s_i) # B, N_token, c_s

        return s_i, z_ij

# Algorithm 18: Sample Diffusion
class SampleDiffusion(nn.Module):
    def __init__(self, c_s, c_z, c_atom, c_atompair, c_token, 
                       gama_0=0.8, gama_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5,
                       sigma_data=16, s_max=160, s_min=4e-4, p=7, N_atoms_max=4096):
        super(SampleDiffusion, self).__init__()

        self.gama_0 = gama_0
        self.gama_min = gama_min
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p

        self.centre_random_augmentation = CentreRandomAugmentation(s_trans=1.0)
        self.diffusion_module = DiffusionModule(c_s=c_s, c_z=c_z, c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, sigma_data=sigma_data, N_atoms_max=N_atoms_max)

    def forward(self, fstar, sinputs_i, strunk_i, ztrunk_ij, T=0):
        # 
        # sinputs_i, strunk_i: B, N_token, c_s
        # ztrunk_ij: B, N_token,B_token, c_z
        N_atom = fstar['ref_pos'].shape[1]
        B,N_token,c_s = sinputs_i.shape

        if T==0:
            # training mode
            ht = self.sigma_data*torch.exp( -1.2 + 1.5*torch.normal(mean=0.0,std=1.0,size=(B,1), device=sinputs_i.device) ) # B, 1; warning! not sure for this part.
            x_l = self.diffusion_module(fstar['atom_coors']+torch.randn_like(fstar['atom_coors'])*ht[:,None],ht,\
                                        fstar,sinputs_i,strunk_i,ztrunk_ij)
        else:
            # prediction mode
            c_i = self.sigma_data * (self.s_max**(1/self.p) + torch.arange(0, 1+1/T, 1/T)*(self.s_min**(1/self.p) - self.s_max**(1/self.p)))**self.p # T

            c_0 = c_i[0]
            x_l = c_0 * torch.normal(mean=0.0,std=1.0, size=(B,N_atom,3), device=sinputs_i.device) # B, N_atom, 3
            for c_1 in c_i[1:]:
                # x_l = self.centre_random_augmentation(x_l) # B, N_atom, 3

                gama = torch.ones((B,1), dtype=sinputs_i.dtype, device=sinputs_i.device)*self.gama_0 # B, 1
                gama[c_1<=self.gama_min] = 0.0 # B, 1

                ht = c_0 * (gama+1) # B, 1
                ksi_l = self.noise_scale_lambda * (torch.sqrt(ht**2-c_0**2) * torch.normal(mean=0.0,std=1.0, size=(B,N_atom,3), device=sinputs_i.device)) # B,N_atom,3

                xnoisy_l = x_l + ksi_l # B, N_atom,3

                xdenoised_l = self.diffusion_module(xnoisy_l, ht, fstar, sinputs_i, strunk_i, ztrunk_ij) # B, N_atom,3

                sigma_l = (x_l - xdenoised_l) / ht # B, N_atom,3

                dt = c_1 - ht # B, 1

                x_l = xnoisy_l + self.step_scale_eta*dt*sigma_l # B, N_atom, 3

                c_0 = c_1

        return x_l,ht # B, N_atom, 3;  B, 1

# Algorithm 19: Centre Random Augmentation
class CentreRandomAugmentation(nn.Module):
    def __init__(self, s_trans=1.0):
        super(CentreRandomAugmentation, self).__init__()

        self.s_trans = s_trans

    def forward(self, x_l):
        # x_l: B, N_atom,3
        B,N_atom,_ = x_l.shape

        x_l = x_l - x_l.mean(axis=1, keepdims=True) # B, N_atom,3
        R = UniformRandomRotation(B).reshape(B,1, 3,3).to(x_l.device) # B, 1,3,3
        t = self.s_trans * torch.normal(mean=0.0, std=1.0, size=(B,1,3), device=x_l.device) # B,1,3

        x_l = (R@x_l[...,None]).squeeze(dim=-1) + t # B, N_atom,3

        return x_l # B, N_atom,3
def UniformRandomRotation(B):
    bi,ci,di = torch.normal(mean=0.0, std=1.0, size=(B,1)),torch.normal(mean=0.0, std=1.0, size=(B,1)),torch.normal(mean=0.0, std=1.0, size=(B,1)) # B,1

    s = torch.sqrt(1+bi**2+ci**2+di**2) # B,1

    ai,bi,ci,di = 1/s,bi/s,ci/s,di/s # B,1
    ai,bi,ci,di = ai[:,None],bi[:,None],ci[:,None],di[:,None] # B,1,1
    ri = torch.cat([torch.cat([ai**2+bi**2-ci**2-di**2, 2*bi*ci-2*ai*di, 2*bi*di+2*ai*ci], dim=2),
                    torch.cat([2*bi*ci+2*ai*di, ai**2-bi**2+ci**2-di**2, 2*ci*di-2*ai*bi], dim=2),
                    torch.cat([2*bi*di-2*ai*ci, 2*ci*di+2*ai*bi, ai**2-bi**2-ci**2+di**2], dim=2)], dim=1) # B,3,3

    return ri

# Algorithm 20: Diffusion Module
class DiffusionModule(nn.Module):
    def __init__(self, c_s, c_z, c_atom=128, c_atompair=16, c_token=768, sigma_data=16, N_atoms_max=4096):
        super(DiffusionModule, self).__init__()

        self.sigma_data = sigma_data

        self.diffusion_conditioning = DiffusionConditioning(c_s=c_s, c_z=c_z, sigma_data=sigma_data)
        self.atom_attention_encoder = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, c_s=c_s, c_z=c_z, N_atoms_max=N_atoms_max)

        self.diffusion_transformer = DiffusionTransformer(c_a=c_token, c_s=c_s, c_z=c_z, N_block=24, N_head=16)

        self.linear_no_bias = LinearNoBias(c_s, c_token, ln=True)
        self.ln = nn.LayerNorm([c_token])

        self.atom_attention_decoder = AtomAttentionDecoder(c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, N_atoms_max=N_atoms_max)

    def forward(self, xnoisy_l, ht, fstar, sinputs_i, strunk_i, ztrunk_ij):
        # xnoisy_l: B, N_atom,3; ht: B,1; sinputs_i: B, N_token,c_s; strunk_i: B, N_token,c_s; ztrunk_ij: B, N_token,N_token,c_z
        
        # Conditioning
        s_i,z_ij = self.diffusion_conditioning(ht, fstar, sinputs_i, strunk_i, ztrunk_ij) # B,N_token,c_s; B,N_token,N_token,c_z

        # Scale positions to dimensionless vectors with approximately unit variance.
        rnoisy_l = xnoisy_l / torch.sqrt(ht**2 + self.sigma_data**2)[:,None] # B, N_atom,3

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a_i,qskip_l,cskip_l,pskip_lm = self.atom_attention_encoder(fstar, rnoisy_l,strunk_i,z_ij) # B,N_token,c_token; B,N_atom,c_atom; B,N_atom,c_atom; B,N_atom,N_atom,c_atompair

        # Full self-attention on token level.
        a_i = a_i + self.linear_no_bias(s_i) # B,N_token,c_token
        a_i = self.diffusion_transformer(a_i, s_i, z_ij, beta_ij=0.0) # B,N_token,c_token
        a_i = self.ln(a_i) # B,N_token,c_token

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        rupdate_l = self.atom_attention_decoder(fstar, a_i, qskip_l, cskip_l, pskip_lm) # B, N_atom, 3

        # Rescale updates to positions and combine with input positions
        xout_l = (self.sigma_data**2/(self.sigma_data**2+ht**2))[:,None] * xnoisy_l + \
                 (self.sigma_data*ht/torch.sqrt(self.sigma_data**2+ht**2))[:,None] * rupdate_l # B, N_atom, 3

        return xout_l

# Algorithm 21: Diffusion Conditioning
class DiffusionConditioning(nn.Module):
    def __init__(self, sigma_data, c_z=128, c_s=384):
        super(DiffusionConditioning, self).__init__()

        self.sigma_data = sigma_data

        self.relative_position_encoding = RelativePositionEncoding(c_z=c_z)

        self.linear_no_bias_1 = LinearNoBias(c_z*2,c_z, ln=True)

        self.transitions_11 = Transition(c=c_z, n=2)
        self.transitions_12 = Transition(c=c_z, n=2)

        self.linear_no_bias_2 = LinearNoBias(c_s*2,c_s, ln=True)
        self.fourier_embedding = FourierEmbedding(c=256)
        self.linear_no_bias_3 = LinearNoBias(256,c_s, ln=True)

        self.transitions_21 = Transition(c=c_s, n=2)
        self.transitions_22 = Transition(c=c_s, n=2)

    def forward(self, ht, fstar, sinputs_i, strunk_i, ztrunk_ij):
        # ht: B,1; sinputs_i: B, N_token,c_s; strunk_i: B, N_token,c_s; ztrunk_ij: B, N_token,N_token,c_z
        B,N_token,c_s = sinputs_i.shape

        # Pair conditioning
        z_ij = torch.cat([ztrunk_ij, self.relative_position_encoding(fstar)], dim=-1) # B, N_token,N_token, c_z*2
        z_ij = self.linear_no_bias_1(z_ij) # B, N_token,N_token, c_z

        z_ij = z_ij+self.transitions_11(z_ij) # B, N_token,N_token, c_z
        z_ij = z_ij+self.transitions_12(z_ij) # B, N_token,N_token, c_z

        # Single conditioning
        s_i = torch.cat([strunk_i, sinputs_i], dim=-1) # B, N_token, c_s*2
        s_i = self.linear_no_bias_2(s_i) # B, N_token,c_s
        n = self.fourier_embedding(1/4 * torch.log(ht/self.sigma_data))[:,None].to(ht.device) # B,1,256
        s_i = s_i+self.linear_no_bias_3(n) # B, N_token,c_s

        s_i = s_i+self.transitions_21(s_i) # B, N_token,c_s
        s_i = s_i+self.transitions_22(s_i) # B, N_token,c_s

        return s_i,z_ij

# Algorithm 22: Fourier Embedding
class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super(FourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1,c)), requires_grad=False) # B,c
        self.b = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1,c)), requires_grad=False) # B,c

        self.c = c

    def forward(self, ht):
        # ht: B,1; 
        # Randomly generate weight/bias once before training
        
        # Compute embeddings
        return torch.cos( 2*torch.pi*(ht*self.w + self.b) ) # B,N_token,c

# Algorithm 23: Diffusion Transformer
class DiffusionTransformer(nn.Module):
    def __init__(self, c_a, c_s, c_z, N_block, N_head):
        super(DiffusionTransformer, self).__init__()

        self.attention_pair_bias_list = nn.ModuleList([AttentionPairBias(c_a, c_s, c_z, N_head) for i in range(N_block)])
        self.conditioned_transition_block_list = nn.ModuleList([ConditionedTransitionBlock(c_a, c_s) for i in range(N_block)])

    def forward(self, a_i, s_i, z_ij, beta_ij):
        # a_i: B, N_token,c_a; s_i: B, N_token,c_s; z_ij: B, N_token,N_token, c_z; beta_ij: B, N_token,N_token

        for attention_pair_bias,conditioned_transition_block in zip(self.attention_pair_bias_list,self.conditioned_transition_block_list):
            b_i = attention_pair_bias(a_i, s_i, z_ij, beta_ij) # B, N_token,c_a;
            a_i = b_i + conditioned_transition_block(a_i, s_i) # B, N_token, c_a

        return a_i # B, N_token, c_a

# Algorithm 24: DiffusionAttention with pair bias and mask
class AttentionPairBias(nn.Module):
    def __init__(self, c_a, c_s, c_z, N_head):
        super(AttentionPairBias, self).__init__()

        self.N_head = N_head
        self.c = c_a//N_head

        if c_s is not None:
            self.ada_ln = AdaLN(c_a, c_s)
        self.ln = nn.LayerNorm([c_a])

        self.linear_1 = nn.Linear(c_a, c_a)
        self.linear_no_bias_1 = LinearNoBias(c_a, 2*c_a)
        self.linear_no_bias_2 = LinearNoBias(c_z, N_head, ln=True)
        self.linear_no_bias_3 = LinearNoBias(c_a, c_a)

        self.linear_no_bias_4 = LinearNoBias(c_a, c_a)

        if c_s is not None:
            self.linear_2 = nn.Linear(c_s, c_a) # biasinit=-2.0

    def forward(self, a_i, s_i, z_ij, beta_ij):
        # a_i: B, N_token,c_a; s_i: B, N_token,c_s; z_ij: B, N_token,N_token, c_z; beta_ij: B, N_token,N_token
        B,N_token,c_a = a_i.shape

        # Input projections
        if s_i is not None:
            a_i = self.ada_ln(a_i, s_i) # B, N_token,c_a
        else:
            a_i = self.ln(a_i) # B, N_token,c_a

        qh_i = self.linear_1(a_i).reshape(B,N_token,self.N_head,self.c).transpose(1,2) # B,N_head, N_token,c
        kvh_i = self.linear_no_bias_1(a_i).reshape(B,N_token,self.N_head,self.c*2).transpose(1,2)
        kh_i,vh_i = kvh_i[...,:self.c],kvh_i[...,self.c:] # B,N_head, N_token,c

        if isinstance(beta_ij, float):
            bh_ij = self.linear_no_bias_2(z_ij).permute(0,3,1,2) + beta_ij # B,N_head, N_token,N_token
        else:
            bh_ij = self.linear_no_bias_2(z_ij).permute(0,3,1,2) + beta_ij[:,None,:N_token,:N_token] # B,N_head, N_token,N_token
        gh_i = F.sigmoid(self.linear_no_bias_3(a_i)).reshape(B,N_token,self.N_head,self.c).transpose(1,2) # B,N_head, N_token,c

        # Attention
        Ah_ij = F.softmax( 1/np.sqrt(self.c)*qh_i@kh_i.transpose(-1,-2)+bh_ij, dim=-1 ) # B,N_head, N_token,N_token

        a_i = gh_i*(Ah_ij@vh_i) # B,N_head, N_token,c
        a_i = self.linear_no_bias_4(a_i.transpose(1,2).reshape(B,N_token,-1)) # B,N_token, c_a

        # Output projection (from adaLN-Zero)
        if s_i is not None:
            a_i = F.sigmoid(self.linear_2(s_i)) * a_i # B,N_token, c_a

        return a_i # B,N_token, c_a

# Algorithm 25: Conditioned Transition Block
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super(ConditionedTransitionBlock, self).__init__()

        self.ada_ln = AdaLN(c_a, c_s)
        self.linear_no_bias_1 = LinearNoBias(c_a, n*c_a)
        self.linear_no_bias_2 = LinearNoBias(c_a, n*c_a)

        self.linear = nn.Linear(c_s, c_a) # biasinit=-2.0

        self.linear_no_bias_3 = LinearNoBias(n*c_a, c_a)
    
    def forward(self, a_i, s_i):
        # a_i: B, N_token,c_a; s_i: B, N_token,c_s;
        a_i = self.ada_ln(a_i, s_i) # B, N_token,c_a
        b_i = F.silu(self.linear_no_bias_1(a_i)) * self.linear_no_bias_2(a_i) # B, N_token,n*c_a

        # Output projection (from adaLN-Zero)
        a_i = F.sigmoid(self.linear(s_i))*self.linear_no_bias_3(b_i) # B, N_token, c_a
        
        return a_i # B, N_token, c_a

# Algorithm 26: Adaptive LayerNorm
class AdaLN(nn.Module):
    def __init__(self, c_a, c_s):
        super(AdaLN, self).__init__()

        self.ln1 = nn.LayerNorm([c_a], elementwise_affine=False, bias=False)
        self.ln2 = nn.LayerNorm([c_s], bias=False)

        self.linear = nn.Linear(c_s, c_a)
        self.linear_no_bias = LinearNoBias(c_s, c_a)

    def forward(self, a_i, s_i):
        # a_i: B, N_token,c_a; s_i: B, N_token,c_s;
        a_i = self.ln1(a_i)
        s_i = self.ln2(s_i)

        a_i = F.sigmoid(self.linear(s_i))*a_i + self.linear_no_bias(s_i)

        return a_i # a_i: B, N_token,c_a; s_i: B, N_token,c_s;

# L_diffusion
class DiffusionLoss(nn.Module):
    def __init__(self, sigma_data, alpha_bond=0.):
        super(DiffusionLoss, self).__init__()
        self.sigma_data = sigma_data # R
        self.alpha_bond = alpha_bond

    def forward(self, xpred_l, xGT_l, ht, w_l=None):
        # xpred_l: B, N_atom,3; xGT_l: B, N_atom,3; ht: B, 1
        B,N_atom,_ = xpred_l.shape

        # w_l: B, N_atom
        if w_l is None:
            w_l = torch.ones((len(xpred_l),N_atom), dtype=xpred_l.dtype, device=xpred_l.device)
        xGT_l_aligned = weighted_rigid_align(xGT_l,xpred_l, w_l=w_l) # B, N_atom,3
        
        # L_MSE
        loss = torch.sum(torch.norm(xpred_l-xGT_l_aligned, dim=-1)*w_l) / torch.sum(w_l) # R

        if self.alpha_bond>0:
            w_l_pair = w_l[:,:,None]*w_l[:,None] # B,N_atom
            L_bond = w_l_pair*(torch.norm(xpred_l[:,:,None]-xpred_l[:,None],dim=-1) - torch.norm(xGT_l[:,:,None]-xGT_l[:,None],dim=-1))**2
            loss += self.alpha_bond* torch.sum(L_bond)/torch.sum(w_l_pair)

        return (ht**2+self.sigma_data**2)/(ht+self.sigma_data)**2 * loss

# Algorithm 27: Smooth LDDT Loss
class SmoothLDDTLoss(nn.Module):
    def __init__(self):
        super(SmoothLDDTLoss, self).__init__()

    def forward(self, x_l, xGT_l, fstar):
        # x_l,xGT_l: B, N_atom, 3
        B,N_atom,_ = x_l.shape

        # Compute distances between all pairs of atoms
        delta_x_lm = torch.sum((x_l[:,:,None] - x_l[:,None])**2, dim=-1) # B, N_atom,N_atom
        delta_xGT_lm = torch.sum((xGT_l[:,:,None] - xGT_l[:,None])**2, dim=-1) # B, N_atom,N_atom

        # Compute distance difference for all pairs of atoms
        delta_lm = torch.abs(delta_xGT_lm - delta_x_lm) # B, N_atom,N_atom
        eps_lm = 1/4 * (F.sigmoid(1/2-delta_lm) + F.sigmoid(1-delta_lm) + F.sigmoid(2-delta_lm) + F.sigmoid(4-delta_lm)) # B, N_atom,N_atom

        # Restrict to bespoke inclusion radius
        fstar['is_nucleotide'] = fstar['is_dna'].bool() | fstar['is_rna'].bool() # B, N_atom
        c_lm = ((delta_xGT_lm<30)|fstar['is_nucleotide'][...,None]) | ((delta_xGT_lm<15)|~fstar['is_nucleotide'][...,None]) # B, N_atom,N_atom

        # avoiding self term
        c_lm &= ~torch.eyes(N_atom, dtype=bool)[None] # B, N_atom,N_atom

        # Compute mean, 
        lddt = eps_lm[c_lm].sum() / c_lm.sum()

        return 1-lddt

# Algorithm 28: Weighted Rigid Align
def weighted_rigid_align(x_l, xGT_l, w_l):
    # x_l,xGT_l: B, N_atom, 3; B, N_atom, 3; w_l: B, N_atom;
    # w_l: B,N_atom; remember to set padding part to zero
    
    # Mean-centre positions
    mu = torch.mean(w_l[...,None]*x_l, dim=1, keepdims=True) / torch.mean(w_l[...,None], dim=1, keepdims=True) # B, N_atom, 3
    muGT = torch.mean(w_l[...,None]*xGT_l, dim=1, keepdims=True) / torch.mean(w_l[...,None], dim=1, keepdims=True) # B, N_atom, 3

    x_l = x_l-mu # B, N_atom, 3
    xGT_l = xGT_l-muGT # B, N_atom, 3

    # Find optimal rotation from singular value decomposition
    U,S,V = torch.svd((w_l[...,None]*xGT_l).transpose(-1,-2)@x_l)
    R = U@V.transpose(-1,-2) # B,3,3

    # Remove reflection
    tmp = torch.det(R)<0
    F = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=R.dtype, device=R.device)[None]
    R[tmp] = U[tmp]@F@V[tmp] # B,3,3
    
    # Apply alignment
    xalign_l = x_l@R.transpose(-1,-2) + muGT # B, N_atom, 3

    return xalign_l.detach() # B, N_atom, 3

# Algorithm 31: Confidence head
class ConfidenceHead(nn.Module):
    def __init__(self, c_s, c_z, b_pae, b_pde, b_plddt, b_distogram, N_block=4):
        super(ConfidenceHead, self).__init__()

        self.linear_no_bias_1 = LinearNoBias(c_s, c_z)
        self.one_hot_d_ij = nn.Embedding.from_pretrained(torch.eye(18), freeze=True)
        self.linear_no_bias_2 = LinearNoBias(18, c_z)
        self.pairformer_stack = PairformerStack(c_s=c_s, c_z=c_z, c=c_z, N_block=N_block)
        self.linear_no_bias_3 = LinearNoBias(c_z, b_pae)
        self.linear_no_bias_4 = LinearNoBias(c_z, b_pde)
        self.linear_no_bias_5 = LinearNoBias(c_s, b_plddt)
        self.linear_no_bias_6 = LinearNoBias(c_s, 2)
        self.linear_no_bias_7 = LinearNoBias(c_z, b_distogram)

    def forward(self, fstar, sinputs_i, s_i, z_ij, xpred_l):
        # sinputs_i: B,N_token,c_s; s_i: B,N_token,c_s; z_ij: B,N_token,N_token,c_z; 
        # xpred_l: B,N_atom,3
        tmp = self.linear_no_bias_1(sinputs_i) # B,N_token,c_z
        z_ij = z_ij + (tmp[:,:,None] + tmp[:,None]) # B,N_token,N_token,c_z

        # Embed pair distances of representative atoms:
        # fstar['is_repr']: B, N_atom;
        # fstar['residue_atom_map']: B, N_residue, N_atom
        xpred_lrepi = (fstar['residue_atom_map']*fstar['is_repr'][:,None])@xpred_l.detach() # B,N_token,3
        d_ij = torch.norm(xpred_lrepi[:,:,None] - xpred_lrepi[:,None], dim=-1) # B,N_token,N_token
        d_ij = ((d_ij-3) // 1.1875).long() # B,N_atom,N_atom
        d_ij[d_ij<0] = -1
        d_ij[d_ij>16] = 16
        d_ij += 1 # B,N_atom,N_atom
        z_ij = z_ij+self.linear_no_bias_2( self.one_hot_d_ij(d_ij) ) # B, N_token,N_token,c_z

        s_i_,z_ij_ = self.pairformer_stack(s_i, z_ij)
        s_i = s_i+s_i_ # B, N_token,c_s
        z_ij = z_ij+z_ij_ # B, N_token,N_token,c_z

        ppae_ij = F.softmax(self.linear_no_bias_3(z_ij), dim=-1) # B,N_token,N_token, b_pae
        ppde_ij = F.softmax(self.linear_no_bias_4(z_ij+z_ij.transpose(-2,-3)), dim=-1) # B,N_token,N_token, b_pde

        s_il = fstar['atom_residue_map'] @ s_i # B,N_atom,c_s
        pplddt_l = F.softmax(self.linear_no_bias_5(s_il), dim=-1) # B,N_atom, b_plddt
        presolved_l = F.softmax(self.linear_no_bias_6(s_il), dim=-1) # B,N_atom, 2

        pdistogram_ij = F.softmax(self.linear_no_bias_7(z_ij), dim=-1) # B,N_token,N_token, b_distogram

        return pplddt_l, ppae_ij, ppde_ij, presolved_l, pdistogram_ij
