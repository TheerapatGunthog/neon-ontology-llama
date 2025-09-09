# รายงานวิเคราะห์ออนโทโลยี: ComputerEducation Subject

## 1.แนวคิด (concept)
- :Subject
- :Computerapplication
- :Language
- :Business
- :Health
- :Creativity
- :Lifeskills

## 2.คุณลักษณะ (property)
- {"name": "rdfs:label", "type": "object", "domain": ":Subject", "range": "xsd:string", "owl_props": []}
- {"name": "rdfs:comment", "type": "object", "domain": ":Subject", "range": "xsd:string", "owl_props": []}
- {"name": "skos:related", "type": "object", "domain": ":Computerapplication", "range": ":Language", "owl_props": []}
- {"name": "skos:broader", "type": "object", "domain": ":Business", "range": ":Language", "owl_props": []}
- {"name": "skos:narrower", "type": "object", "domain": ":Health", "range": ":Lifeskills", "owl_props": []}

## 3.ความสัมพันธ์ (relationship)

### 3.1) ความสัมพันธ์แบบลำดับชั้น (subclass)
- [":Computerapplication", ":Subject"]
- [":Language", ":Subject"]
- [":Business", ":Subject"]
- [":Health", ":Subject"]
- [":Creativity", ":Subject"]
- [":Lifeskills", ":Subject"]

### 3.2) ความสัมพันธ์แบบเป็นส่วนหนึ่ง (part-of)
- ไม่มี

### 3.3) ความสัมพันธ์เชิงความหมาย (syn-of)
- ไม่มี

### 3.4) ความสัมพันธ์การเป็นตัวแทน (instance-of)
- ไม่มี

## 4.ข้อกำหนดในการสร้างความสัมพันธ์ (axiom)
- Infer subclass from rdfs:subClassOf.
- Infer instance-of from rdf:type of subclasses.
- Propose part-of only if composition is implied by labels/comments/SKOS.
- Propose syn-of for clear synonymy; avoid speculation.
- List properties if any custom predicates are used; otherwise suggest needed ones.
- Propose axioms like disjointness, transitivity, symmetry, domain/range as recommendations.
