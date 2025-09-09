# รายงานวิเคราะห์ออนโทโลยี: ComputerIndustry Skill

## 1.แนวคิด (concept)
- :Skill
- :Programminglanguage
- :Dataanalysis
- :Software
- :Cloudcomputing
- :Artificialintelligence
- :Datascience
- :Timemanagement
- :Microsoftoffice
- :Tableau
- :Certifiedinternalauditor

## 2.คุณลักษณะ (property)
- {"name": "rdfs:label", "type": "object", "domain": ":Skill", "range": "xsd:string", "owl_props": []}
- {"name": "rdfs:comment", "type": "object", "domain": ":Skill", "range": "xsd:string", "owl_props": []}
- {"name": "skos:related", "type": "object", "domain": ":Programminglanguage", "range": ":Dataanalysis", "owl_props": []}
- {"name": "skos:broader", "type": "object", "domain": ":Dataanalysis", "range": ":Software", "owl_props": []}
- {"name": "skos:related", "type": "object", "domain": ":Cloudcomputing", "range": ":Datascience", "owl_props": []}
- {"name": "skos:broader", "type": "object", "domain": ":Artificialintelligence", "range": ":Datascience", "owl_props": []}
- {"name": "skos:related", "type": "object", "domain": ":Datascience", "range": ":Timemanagement", "owl_props": []}
- {"name": "skos:related", "type": "object", "domain": ":Microsoftoffice", "range": ":Tableau", "owl_props": []}
- {"name": "skos:broader", "type": "object", "domain": ":Certifiedinternalauditor", "range": ":Dataanalysis", "owl_props": []}

## 3.ความสัมพันธ์ (relationship)

### 3.1) ความสัมพันธ์แบบลำดับชั้น (subclass)
- [":Skill", ":Programminglanguage"]
- [":Skill", ":Dataanalysis"]
- [":Skill", ":Software"]
- [":Skill", ":Cloudcomputing"]
- [":Skill", ":Artificialintelligence"]
- [":Skill", ":Datascience"]
- [":Skill", ":Timemanagement"]
- [":Skill", ":Microsoftoffice"]
- [":Skill", ":Tableau"]
- [":Skill", ":Certifiedinternalauditor"]

### 3.2) ความสัมพันธ์แบบเป็นส่วนหนึ่ง (part-of)
- ไม่มี

### 3.3) ความสัมพันธ์เชิงความหมาย (syn-of)
- [":Programminglanguage", ":Dataanalysis"]
- [":Cloudcomputing", ":Datascience"]
- [":Artificialintelligence", ":Datascience"]
- [":Datascience", ":Timemanagement"]
- [":Microsoftoffice", ":Tableau"]
- [":Certifiedinternalauditor", ":Dataanalysis"]

### 3.4) ความสัมพันธ์การเป็นตัวแทน (instance-of)
- ไม่มี

## 4.ข้อกำหนดในการสร้างความสัมพันธ์ (axiom)
- Infer subclass from rdfs:subClassOf.
- Infer instance-of from rdf:type of subclasses.
- Propose part-of only if composition is implied by labels/comments/SKOS.
- Propose syn-of for clear synonymy; avoid speculation.
- List properties if any custom predicates are used; otherwise suggest needed ones.
- Propose axioms like disjointness, transitivity, symmetry, domain/range as recommendations.
