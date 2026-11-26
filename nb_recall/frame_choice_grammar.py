"""Model selection within annotated memory candidates; does not claim blind retrieval."""
import json


class FrameChoiceGrammar:
    def __init__(self, tokenizer, groups):
        self.eos=tokenizer.eos_token_id
        if self.eos is None:raise ValueError('Tokenizer must define EOS')
        segments=[['{"decision":"ANSWER","frames":[']]
        for i,group in enumerate(groups):
            if not isinstance(group,list) or not all(type(v)==int and -1<=v<96 for v in group):raise ValueError('Invalid annotated frame group')
            if -1 in group and group!=[-1]:raise ValueError('Sentinel cannot be mixed with image indices')
            choices=[json.dumps([v],separators=(',',':')) for v in dict.fromkeys(group)] if group else ['[]']
            if i<len(groups)-1:choices=[x+',' for x in choices]
            segments.append(choices)
        segments.append(['],"observe_frames":[]}'])
        self.roots=[];self.max_tokens=0
        for choices in segments:
            root={};longest=0
            for choice in choices:
                tokens=tokenizer.encode(choice,add_special_tokens=False);longest=max(longest,len(tokens));node=root
                for token in tokens:node=node.setdefault(token,{})
                node[None]=True
            self.roots.append(root);self.max_tokens+=longest

    def allowed(self,tokens):
        segment=0;node=self.roots[0]
        for token in tokens:
            while None in node and len(node)==1:
                segment+=1
                if segment==len(self.roots):return [self.eos]
                node=self.roots[segment]
            if token not in node:raise ValueError('Generated prefix escaped the memory candidate grammar')
            node=node[token]
        while None in node and len(node)==1:
            segment+=1
            if segment==len(self.roots):return [self.eos]
            node=self.roots[segment]
        return [token for token in node if token is not None]
