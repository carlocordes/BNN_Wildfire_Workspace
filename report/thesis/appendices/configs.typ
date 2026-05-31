#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

= Experiment Configs

== Experiment 1: Sequence Extent

=== config_t009_1
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4 
    sample_extent : 3
    sample_period : 1 
    target_extent : 14
    lead_time : 0 
```

=== config_t009_2
#codly(languages: codly-languages)
```yaml
temporal_extent:
    day_interval: 1
    sequence_period : 4 
    sample_extent : 5 
    sample_period : 1 
    target_extent : 14
    lead_time : 0 
```

=== config_t009_3
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4 
    sample_extent : 7
    sample_period : 1 
    target_extent : 14
    lead_time : 0 
```

=== config_t009_4
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4
    sample_extent : 1 
    sample_period : 1
    target_extent : 14
    lead_time : 0 
```
#pagebreak()


== Experiment 2: Lead-Time
=== config_t010_1
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4
    sample_extent : 5
    sample_period : 1
    target_extent : 14
    lead_time : -5 
```

=== config_t010_2
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4 #
    sample_extent : 5 
    sample_period : 1 
    target_extent : 14
    lead_time : 5 
```

=== config_t010_1
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4
    sample_extent : 5
    sample_period : 1 
    target_extent : 14
    lead_time : 10 

```

=== config_t010_4
#codly(languages: codly-languages)
```yaml
  temporal_extent:
    day_interval: 1
    sequence_period : 4 
    sample_extent : 5 
    sample_period : 1
    target_extent : 14
    lead_time : 20 
```